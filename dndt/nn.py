import math
import pickle
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow_addons as tfa

from .util import normalize

for x in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(x, True)


def get_dnn_model(
        inputs,
        layer_type='rnn',
        n_layers=1,
        n_units=128,
        n_projection_layers=1,
        kernel_width=20,
        inner_activation='gelu',
        n_outputs=300,
        dropout=None,
        input_dropout=None,
        temporal_dropout=None,
        reg_scale=1.,
        sensor_filter_scale=None,
        use_glove=False,
        use_resnet=False,
        use_locally_connected=False,
        use_time_mask=False,
        time_mask_regularizer_scale=None,
        independent_channels=False,
        batch_normalize=False,
        layer_normalize=False,
        l2_layer_normalize=False,
        variational=False,
        n_train=1
):
    if use_glove:
        output_activation = None
    else:
        output_activation = 'softmax'

    if reg_scale:
        kernel_regularizer = tf.keras.regularizers.L2(reg_scale)
    else:
        kernel_regularizer = None

    noise_shape = [1, 1, None]

    layers = []
    if sensor_filter_scale:
        layers.append(SensorFilter(rate=sensor_filter_scale))
    if input_dropout:
        layers.append(tf.keras.layers.Dropout(input_dropout, noise_shape=noise_shape))
    if temporal_dropout:
        layers.append(tf.keras.layers.Dropout(temporal_dropout, noise_shape=inputs.shape[:-1] + [1]))
    if use_locally_connected:
        layers.append(tf.keras.layers.ZeroPadding1D(padding=(kernel_width - 1, 0)))

    if independent_channels:
        assert not variational, 'Independent channels not currently supported using variational inference'
        groups = inputs.shape[-1]
    else:
        groups = 1

    if use_resnet:
        layers.append(
            tf.keras.layers.Dense(
                n_units,
                kernel_regularizer=kernel_regularizer,
                activation=inner_activation,
                groups=groups
            )
        )
        if batch_normalize:
            layers.append(tf.keras.layers.BatchNormalization(center=False, scale=False))
        if layer_normalize:
            layers.append(tf.keras.layers.LayerNormalization(center=False, scale=False))
        if l2_layer_normalize:
            layers.append(L2LayerNormalization())
        if dropout:
            layers.append(tf.keras.layers.Dropout(dropout, noise_shape=noise_shape))
    for _ in range(n_layers):
        if layer_type == 'dense':
            assert not variational, 'Variational inference in Dense networks not currently supported'
            layers.append(
                tf.keras.layers.Dense(
                    n_units,
                    kernel_regularizer=kernel_regularizer,
                    activation=inner_activation
                )
            )
        elif layer_type == 'cnn':
            if use_resnet:
                assert not variational, 'Variational inference in Resnets not currently supported'
                layers.append(
                    ResNetConv1DBlock(
                        kernel_width,
                        reg_scale=reg_scale,
                        inner_activation=inner_activation,
                        activation=None,
                        layer_normalize=layer_normalize,
                        batch_normalize=batch_normalize
                    )
                )
            else:
                if variational:
                    layers.append(tf.keras.layers.ZeroPadding1D(padding=(kernel_width - 1, 0)))
                    layers.append(
                        Convolution1DFlipout(
                            n_units,
                            kernel_width,
                            padding='valid',
                            activation=inner_activation,
                            n_train=n_train
                        )
                    )
                else:
                    layers.append(
                        tf.keras.layers.Conv1D(
                            n_units,
                            kernel_width,
                            padding='causal',
                            kernel_regularizer=kernel_regularizer,
                            activation=inner_activation,
                            groups=groups
                        )
                    )
        elif layer_type == 'rnn':
            assert not variational, 'Variational inference in RNNs not currently supported'
            layers.append(
                tf.keras.layers.LSTM(
                    n_units,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=kernel_regularizer,
                    return_sequences=True
                )
            )
        else:
            raise ValueError('Unrecognized layer type: %s' % layer_type)
        if batch_normalize:
            layers.append(tf.keras.layers.BatchNormalization(center=False, scale=False))
        if layer_normalize:
            layers.append(tf.keras.layers.LayerNormalization(center=False, scale=False))
        if l2_layer_normalize:
            layers.append(L2LayerNormalization())
        if dropout:
            layers.append(tf.keras.layers.Dropout(dropout, noise_shape=noise_shape))
    if use_locally_connected:
        assert not variational, 'Variational inference in locally connected layers not currently supported'
        if n_projection_layers:
            lc_units = n_units
            lc_activation = inner_activation
            lc_regularizer = kernel_regularizer
        else:
            lc_units = n_outputs
            lc_activation = output_activation
            lc_regularizer = None
        layers.append(
            tf.keras.layers.LocallyConnected1D(
                lc_units,
                kernel_width,
                padding='valid',
                kernel_regularizer=lc_regularizer,
                activation=lc_activation,
                implementation=1
            )
        )
        if n_projection_layers:
            if batch_normalize:
                layers.append(tf.keras.layers.BatchNormalization(center=False, scale=False))
            if layer_normalize:
                layers.append(tf.keras.layers.LayerNormalization(center=False, scale=False))
            if l2_layer_normalize:
                layers.append(L2LayerNormalization())
            if dropout:
                layers.append(tf.keras.layers.Dropout(dropout, noise_shape=noise_shape))
    for i in range(n_projection_layers):
        if variational:
            layers.append(
                DenseFlipout(
                    n_outputs,
                    activation=output_activation,
                    n_train=n_train
                )
            )
        else:
            layers.append(
                tf.keras.layers.Dense(
                    n_outputs,
                    kernel_regularizer=kernel_regularizer,
                    activation=output_activation
                )
            )
        if i < n_projection_layers - 1:
            if batch_normalize:
                layers.append(tf.keras.layers.BatchNormalization(center=False, scale=False))
            if layer_normalize:
                layers.append(tf.keras.layers.LayerNormalization(center=False, scale=False))
            if l2_layer_normalize:
                layers.append(L2LayerNormalization())
            if dropout:
                layers.append(tf.keras.layers.Dropout(dropout, noise_shape=noise_shape))

    if use_time_mask:
        layers.append(TrainableTimeMask(rate=time_mask_regularizer_scale))

    outputs = inputs
    for layer in layers:
        outputs = layer(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def eval_and_save(
        model_path,
        eval_data_generator,
        ylab,
        results_path,
        use_glove=False,
        ix2lab=None,
        comparison_set=None,
        results_dict=None
):
    if results_dict is None:
        results_dict = {}
    model = tf.keras.models.load_model(model_path)
    y_pred = dnn_classify(
        model,
        eval_data_generator,
        use_glove=use_glove,
        ix2lab=ix2lab,
        comparison_set=comparison_set
    )
    T = y_pred.shape[1]

    results_acc = np.zeros(T)
    results_f1 = np.zeros(T)
    for t in range(T):
        acc = accuracy_score(ylab, y_pred[:, t])
        f1 = f1_score(ylab, y_pred[:, t], average='macro')
        results_acc[t] = acc
        results_f1[t] = f1
    results_dict['acc'] = results_acc
    results_dict['f1'] = results_f1
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)


def dnn_classify(
        model,
        X,
        use_glove=False,
        ix2lab=None,
        comparison_set=None,
        **kwargs
):
    if use_glove:
        assert comparison_set is not None, 'Classification using GloVe requires a comparison set'
    if ix2lab is None:
        ix2lab = {}

    outputs = []
    for _X in X:
        _X = _X[0]
        _outputs = model.predict_on_batch(_X, **kwargs)
        outputs.append(_outputs)
    outputs = np.concatenate(outputs, axis=0)

    if use_glove:
        outputs = normalize(outputs, axis=-1)
        classes = np.array(sorted(list(comparison_set.keys())))
        glove_targ = np.stack([comparison_set[x] for x in classes], axis=1)
        glove_targ = normalize(glove_targ, axis=0)

        outputs = np.dot(outputs, glove_targ)
        ix = np.argmax(outputs, axis=-1)
        pred = classes[ix]
    else:
        if comparison_set:
            mask = np.zeros(outputs.shape[-1])
            mask[comparison_set] = 1
            while len(mask.shape) < len(outputs.shape):
                mask = mask[None, ...]
            outputs *= mask
        pred = np.argmax(outputs, axis=-1)
        pred = np.vectorize(lambda x: ix2lab.get(x, '<<OOV>>'))(pred)

    return pred


class ModelEval(tf.keras.callbacks.Callback):
    def __init__(
            self,
            model_path,
            eval_freq=10,
            eval_data_generator=None,
            ylab=None,
            use_glove=False,
            ix2lab=None,
            comparison_set=None,
            results_dict=None,
            results_path=None
    ):
        self.model_path = model_path
        self.eval_freq = eval_freq
        self.eval_data_generator = eval_data_generator
        self.ylab = ylab
        self.use_glove = use_glove
        self.ix2lab = ix2lab
        self.comparison_set = comparison_set
        self.results_dict = results_dict
        self.results_path = results_path

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.eval_freq == 0 and self.eval_data_generator is not None:
            eval_and_save(
                self.model_path,
                self.eval_data_generator,
                self.ylab,
                self.results_path,
                use_glove=self.use_glove,
                ix2lab=self.ix2lab,
                comparison_set=self.comparison_set,
                results_dict=self.results_dict
            )


class ModelCheckpoint(tfa.callbacks.AverageModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.save_freq.startswith('epoch'):
            if self.save_freq == 'epoch':
                m = 1
            else:
                m = int(self.save_freq[5:])
            if (epoch + 1) % m == 0:
                self._save_model(epoch=epoch, batch=None, logs=logs)


class RasterData(tf.keras.utils.Sequence):
    def __init__(
            self,
            x,
            y=None,
            sample_weights=None,
            batch_size=128,
            shuffle=False,
            contrastive_sampling=False
    ):
        self.x = np.array(x)
        if y is None:
            self.y = None
        else:
            self.y = np.array(y)
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.contrastive_sampling = contrastive_sampling
        if self.contrastive_sampling:
            y_uniq, y_ix_inv = np.unique(y, return_inverse=True, axis=0)
            n_y_uniq = len(y_uniq)
            self.y_uniq = y_uniq
            self.y_ix_inv = y_ix_inv
            self.n_y_uniq = n_y_uniq

        self.set_idx()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.ix[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[indices]
        if self.y is not None:
            batch_y = self.y[indices]
            tile_ix = [1, batch_x.shape[1]]
            if len(self.y.shape) == 2:
                tile_ix.append(1)
            batch_y = np.tile(np.expand_dims(batch_y, axis=1), tile_ix)
            if self.contrastive_sampling:
                y_ix_inv = self.y_ix_inv[indices]
                contrastive_ix = np.mod(y_ix_inv + np.random.randint(1, self.n_y_uniq), self.n_y_uniq)
                contrastive_targets = self.y_uniq[contrastive_ix]
                contrastive_targets = np.tile(np.expand_dims(contrastive_targets, axis=1), tile_ix)
                batch_y = (batch_y, contrastive_targets)
        else:
            batch_y = None
        if self.sample_weights is not None:
            batch_sample_weights = self.sample_weights
        else:
            batch_sample_weights = None

        return (batch_x, batch_y, batch_sample_weights)

    def on_epoch_end(self):
        self.set_idx()

    def set_idx(self):
        ix = np.arange(len(self.x))
        if self.shuffle:
            ix = np.random.permutation(ix)
        self.ix = ix


@tf.keras.utils.register_keras_serializable()
class Convolution1DFlipout(tfp.layers.Convolution1DFlipout):
    def __init__(
            self,
            filters,
            kernel_size,
            strides=1,
            padding='valid',
            activation=None,
            n_train=1
    ):
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                  tf.cast(n_train, dtype=tf.float32))
        super(Convolution1DFlipout, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_divergence_fn=kl_divergence_function
        )
        self._meta = {
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides,
            'activation': activation,
            'n_train': n_train
        }

    def get_config(self):
        return self._meta.copy()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DenseFlipout(tfp.layers.DenseFlipout):
    def __init__(
            self,
            units,
            activation=None,
            n_train=1
    ):
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                  tf.cast(n_train, dtype=tf.float32))
        super(DenseFlipout, self).__init__(
            units,
            activation=activation,
            kernel_prior_fn=None,
            kernel_divergence_fn=kl_divergence_function
        )
        self._meta = {
            'units': units,
            'activation': activation,
            'n_train': n_train
        }

    def get_config(self):
        return self._meta.copy()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ResNetConv1DBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_width,
            n_layers=2,
            reg_scale=None,
            inner_activation=None,
            activation=None,
            layer_normalize=False,
            batch_normalize=False,
            **kwargs
    ):
        super(ResNetConv1DBlock, self).__init__(**kwargs)
        self.kernel_width = kernel_width
        self.n_layers = n_layers
        self.reg_scale = reg_scale
        self.inner_activation = inner_activation
        self.activation = activation
        self.layer_normalize = layer_normalize
        self.batch_normalize = batch_normalize

        if self.reg_scale:
            kernel_regularizer = tf.keras.regularizers.L2(self.reg_scale)
        else:
            kernel_regularizer = None
        self.kernel_regularizer = kernel_regularizer


    def build(self, input_shape):
        _x = tf.keras.Input(input_shape)
        n_units = _x.shape[-1]
        _layers = []
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                activation = self.inner_activation
            else:
                activation = self.activation
            _layers.append(
                tf.keras.layers.Conv1D(
                    n_units,
                    self.kernel_width,
                    padding='causal',
                    kernel_regularizer=self.kernel_regularizer,
                    activation=activation
                )
            )
            if self.batch_normalize:
                _layers.append(tf.keras.layers.BatchNormalization())
            if self.layer_normalize:
                _layers.append(tf.keras.layers.LayerNormalization())

        self._layers = _layers

        self._add = tf.keras.layers.Add()

        self.built = True

    def call(self, inputs, training=False):
        _x = inputs
        x = inputs
        for layer in self._layers:
            x = layer(x, training=training)

        x = self._add([_x, x])

        return x

    def get_config(self):
        config = super(ResNetConv1DBlock, self).get_config()
        config.update({
            'kernel_width': self.kernel_width,
            'n_layers': self.n_layers,
            'reg_scale': self.reg_scale,
            'inner_activation': self.inner_activation,
            'activation': self.activation,
            'layer_normalize': self.layer_normalize,
            'batch_normalize': self.batch_normalize,
        })

        return config


@tf.keras.utils.register_keras_serializable()
class L2LayerNormalization(tf.keras.layers.Layer):
    def __init__(
            self,
            epsilon=0.001,
            center=True,
            scale=True,
            **kwargs
    ):
        super(L2LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        ndim = int(input_shape[-1])
        self.gamma = self.add_weight(
            name='gamma',
            shape=(ndim,),
            initializer='ones'
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(ndim,),
            initializer='zeros'
        )

        self.built = True

    def call(self, inputs, training=False):
        norm = tf.maximum(tf.linalg.norm(inputs, axis=-1, keepdims=True), self.epsilon)
        gamma = self.gamma
        while len(gamma.shape) < len(inputs.shape):
            gamma = gamma[None, ...]
        beta = self.beta
        while len(beta.shape) < len(inputs.shape):
            beta = beta[None, ...]
        return inputs / norm * gamma + beta

    def get_config(self):
        config = super(L2LayerNormalization, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
        })

        return config


@tf.keras.utils.register_keras_serializable()
class SensorFilter(tf.keras.layers.Layer):
    def __init__(
            self,
            rate=None,
            **kwargs
    ):
        super(SensorFilter, self).__init__(**kwargs)

        self.rate = rate
        if self.rate:
            self.w_regularizer = tf.keras.regularizers.L1(self.rate)
        else:
            self.w_regularizer = None

    def build(self, input_shape):
        ndim = int(input_shape[-1])
        self.w = self.add_weight(
            name='filter_weights',
            shape=(ndim,),
            initializer='ones',
            regularizer=self.w_regularizer
        )

        self.built = True

    def call(self, inputs, training=False):
        x = inputs
        w = self.w
        w = tf.tanh(w)
        while len(w.shape) < len(x.shape):
            w = w[None, ...]
        x = x * w

        return x

    def get_config(self):
        config = super(SensorFilter, self).get_config()
        config.update({
            'rate': self.rate
        })

        return config


@tf.keras.utils.register_keras_serializable()
class TrainableTimeMask(tf.keras.layers.Layer):
    def __init__(
            self,
            rate=None,
            **kwargs
    ):
        super(TrainableTimeMask, self).__init__(**kwargs)

        self.rate = rate

    def build(self, input_shape):
        ndim = int(input_shape[-2])
        if self.rate:
            regularizer = tf.keras.regularizers.L2(self.rate / ndim)
        else:
            regularizer = None
        self.w = self.add_weight(
            name='time_mask',
            shape=(ndim,),
            initializer='zeros',
            regularizer=regularizer
        )

        self.built = True

    def call(self, inputs, training=False):
        # tf.print('w', tf.squeeze(self.w), 'attn', tf.nn.softmax(self.w), summarize=20)
        return inputs

    def compute_mask(self, inputs, mask=None):
        input_shape = inputs.shape
        attn = tf.nn.softmax(self.w)[..., None] # Time must be 2nd to last dim
        while len(attn.shape) < len(input_shape):
            attn = attn[None, ...]
        ntime = tf.cast(tf.shape(self.w)[0], dtype=tf.float32)
        return attn * ntime

    def get_config(self):
        config = super(TrainableTimeMask, self).get_config()
        config.update({
            'rate': self.rate
        })

        return config
