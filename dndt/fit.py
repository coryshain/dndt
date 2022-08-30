import os
import math
import shutil
import pickle
import re
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.pipeline import Pipeline
import argparse
import tensorflow_addons as tfa

from .util import *
from .nn import *

channel_matcher = re.compile('(MEG\d\d\d\d)')

MAX_N_RESAMP = 100


glove_matcher = re.compile('d\d{3}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run cross-validated decoding.
    ''')
    argparser.add_argument('config', help='Path to config file containing decoding settings.')
    argparser.add_argument('iteration', type=int, help='Which CV iteration to run (1-indexed).')
    argparser.add_argument('fold', type=int, help='Which CV fold to run (1-indexed).')
    argparser.add_argument('-s', '--save_freq', type=int, default=10, help='Save frequency (in epochs).')
    argparser.add_argument('-f', '--force_restart', action='store_true', help='Force model training from initialization, even if a saved model exists.')
    argparser.add_argument('-F', '--force_resample_cv', action='store_true', help='Force resampling of CV partition, even if a saved partition exists.')
    argparser.add_argument('-c', '--cpu_only', action='store_true', help='Force CPU implementation if GPU available.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    force_restart = args.force_restart
    force_resample_cv = args.force_resample_cv
    save_freq = args.save_freq
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    paths = config['paths']
    label_field = config.get('label_field')
    nfolds = config.get('nfolds', 10)
    assert nfolds > 1, "nfolds must be >= 2."
    niter = config.get('niter', 1)
    filters = config.get('filters', {})
    powerband = config.get('powerband', None)
    downsample_by = config.get('downsample_by', 10)
    use_glove = config.get('use_glove', False)
    zscore_time = config.get('zscore_time', True)
    zscore_sensors = config.get('zscore_sensors', False)
    normalize_sensors = config.get('normalize_sensors', False)
    tanh_transform = config.get('tanh_transform', False)
    rank_transform_sensors = config.get('rank_transform_sensors', False)
    k_feats = config.get('k_feats', 0)
    k_pca_glove = config.get('k_pca_glove', 0)
    k_pca_drop = config.get('k_pca_drop', 0)
    layer_type = config.get('layer_type', 'cnn')
    learning_rate = config.get('learning_rate', 0.0001)
    n_units = config.get('n_units', 64)
    n_layers = config.get('n_layers', 2)
    kernel_width = config.get('kernel_width', 10)
    cnn_activation = config.get('cnn_activation', 'relu')
    reg_scale = config.get('reg_scale', 0.0001)
    dropout = config.get('dropout', 0.1)
    input_dropout = config.get('input_dropout', None)
    temporal_dropout = config.get('temporal_dropout', None)
    use_resnet = config.get('use_resnet', False)
    use_locally_connected = config.get('use_locally_connected', False)
    use_sensor_mask = config.get('use_sensor_mask', False)
    sensor_mask_regularizer_scale = config.get('sensor_mask_regularizer_scale', 0.1)
    use_time_mask = config.get('use_time_mask', False)
    time_mask_regularizer_scale = config.get('time_mask_regularizer_scale', 0.1)
    independent_channels = config.get('independent_channels', False)
    batch_normalize = config.get('batch_normalize', True)
    layer_normalize = config.get('layer_normalize', False)
    l2_layer_normalize = config.get('l2_layer_normalize', False)
    n_projection_layers = config.get('n_projection_layers', 1)
    variational = config.get('variational', False)
    contrastive_loss_weight = config.get('contrastive_loss_weight', None)
    index_subjects = config.get('index_subjects', True)
    n_dnn_epochs = config.get('n_dnn_epochs', 500)
    dnn_batch_size = config.get('dnn_batch_size', 32)
    inner_validation_split = config.get('inner_validation_split', None)
    score = config.get('score', 'acc')
    confint = config.get('confint', 95)
    onset = config.get('onset', 0.5)
    outdir = config.get('outdir', './results')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.normpath(os.path.realpath(config_path)) == os.path.normpath(os.path.realpath(outdir + '/config.ini')):
        shutil.copy2(config_path, outdir + '/config.yml')

    X_train = []
    y_train = []
    ntime = None
    iteration = args.iteration
    fold = args.fold
    n_subject = len(paths)
    subject_map = {}
    for i, dirpath in enumerate(paths):
        dirpath = os.path.normpath(dirpath)
        stderr('Loading %s...\n' % dirpath)
        filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
        cache_path = os.path.join(dirpath, filename)
        if not os.path.exists(cache_path):
            compile_data(
                dirpath,
                downsample_by=downsample_by,
                powerband=powerband
            )
        with open(cache_path, 'rb') as f:
            data_src = pickle.load(f)
        _data = data_src['data']
        _labels = data_src['labels']
        meta = data_src['meta']

        cols = [label_field]
        if use_glove:
            cols += sorted([x for x in _labels if glove_matcher.match(x)])

        _label_df = pd.DataFrame(_labels)
        if 'labcount' not in _label_df.columns:
            labs, ix, counts = np.unique(_label_df[label_field].values, return_inverse=True, return_counts=True)
            counts = counts[ix]
            _label_df['labcount'] = counts
        cols = [label_field]
        if use_glove:
            cols += sorted([x for x in _labels if glove_matcher.match(x)])
        _labels = np.stack([_labels[x] for x in cols], axis=1)

        if zscore_time:
            stderr('Z-scoring over time...\n')
            _data = zscore(_data, axis=2)
        if zscore_sensors:
            stderr('Z-scoring over sensors...\n')
            _data = zscore(_data, axis=1)
        if tanh_transform:
            stderr('Tanh-transforming...\n')
            _data = np.tanh(_data)
        if normalize_sensors:
            stderr('L2 normalizing over sensors...\n')
            n = np.linalg.norm(_data, axis=1, keepdims=True)
            _data /= n
        if rank_transform_sensors:
            stderr('Rank-transforming over sensors...\n')
            _ndim = _data.shape[1]
            _mean = (_ndim + 1) / 2
            _sd = np.arange(_ndim).std()
            _data = (rankdata(_data, axis=1) - 1) / _sd

        _data = np.where(np.isfinite(_data), _data, 0.)
        _data = np.transpose(_data, [0, 2, 1])

        if index_subjects:
            subject_ix = np.zeros(_data.shape[:-1] + (n_subject,))
            subject_ix[..., i] = 1
            _data = np.concatenate([_data, subject_ix], axis=-1)
            subject_map[dirpath] = i

        train_ix_path = os.path.join(dirpath, 'train_ix_i%d_f%d.obj' % (iteration, fold))
        if force_resample_cv or not os.path.exists(train_ix_path):
            # Risky if run in parallel, since multiple diff runs could resample the entire partition at the same time
            # Recommended to compile CV first using meg_coref.compile_cv_ix
            compile_cv_ix(_data, dirpath, niter=niter, nfolds=nfolds)
        with open(train_ix_path, 'rb') as f:
            train_ix = pickle.load(f)

        _X_train = _data[train_ix]
        _y_train = _labels[train_ix]

        _filter_mask = compute_filter_mask(_label_df.iloc[train_ix], filters)
        _X_train = _X_train[_filter_mask]
        _y_train = _y_train[_filter_mask]

        X_train.append(_X_train)
        y_train.append(_y_train)

        if ntime is None:
            ntime = _X_train.shape[1]

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    if inner_validation_split:
        n_train = int(len(X_train) * (1 - inner_validation_split))
    else:
        n_train = len(X_train)
    if use_glove and k_pca_glove:
        stderr('Selecting top %d principal GloVe components...\n' % k_pca_glove)
        vocab, ix = np.unique(y_train[:, 0], return_index=True)
        _pca_in = y_train[ix, 1:]
        glove_pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA(k_pca_glove))])
        glove_pca.fit(_pca_in)
    else:
        glove_pca = None

    stderr('Regressing model...\n')
    labs, counts = np.unique(y_train[:, 0], return_counts=True)

    _nclass = len(labs)
    _maj_class = labs[counts.argmax()]
    _maj_class_pred = np.array([_maj_class])
    _baseline_preds = np.tile(_maj_class_pred, [len(y_train)])
    chance_score_acc = accuracy_score(y_train[:, 0], _baseline_preds)
    _probs = counts / counts.sum()
    _baseline_preds = np.random.multinomial(1, _probs, size=len(y_train))
    _baseline_preds = _baseline_preds.argmax(axis=1)
    _baseline_preds = labs[_baseline_preds]
    chance_score_f1 = f1_score(y_train[:, 0], _baseline_preds, average='macro')
    if use_glove and k_pca_glove:
        stderr('PCA-transforming GloVe components...\n')
        y_train = np.concatenate([y_train[:, :1], glove_pca.transform(y_train[:, 1:])], axis=1)

    stderr('Iteration %d, CV fold %d\n' % (iteration, fold))
    fold_path = os.path.join(os.path.normpath(outdir), 'i%d_f%d' % (iteration, fold))
    model_path = os.path.join(fold_path, 'model.h5')
    ema_path = os.path.join(fold_path, 'model_ema.h5')
    tb_path = os.path.join(fold_path, 'tensorboard')
    results_path = os.path.join(fold_path, 'results.obj')
    metadata_path = os.path.join(fold_path, 'metadata.obj')
    summary_path = os.path.join(fold_path, 'summary.txt')

    y_train_lab = y_train[:, 0]
    y_lab_uniq, y_lab_counts = np.unique(y_train_lab, return_counts=True)
    lab2ix = {_y: i for i, _y in enumerate(y_lab_uniq)}
    ix2lab = {lab2ix[_y]: _y for _y in lab2ix}
    y_train_lab_ix = np.vectorize(lab2ix.__getitem__)(y_train_lab)
    if use_glove:
        y_train_glove = y_train[:, 1:].astype('float32')
        comparison_set = {}
        y_train = y_train_glove
        n_outputs = y_train_glove.shape[1]
        monitor = 'val_sim'
    else:
        y_train_glove = None
        comparison_set = None
        y_train = y_train_lab_ix
        n_outputs = len(lab2ix)
        monitor = 'val_acc'

    if not force_restart and os.path.exists(model_path):
        stderr('Loading saved checkpoint...\n')
        m = tf.keras.models.load_model(model_path)
        inputs = None
        optimizer = None
    else:
        if force_restart and os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        inputs = tf.keras.Input(
            shape=list(X_train.shape[1:])
        )

        m = get_dnn_model(
            inputs,
            layer_type=layer_type,
            n_units=n_units,
            n_layers=n_layers,
            kernel_width=kernel_width,
            inner_activation=cnn_activation,
            n_outputs=n_outputs,
            reg_scale=reg_scale,
            dropout=dropout,
            input_dropout=input_dropout,
            temporal_dropout=temporal_dropout,
            use_glove=use_glove,
            use_resnet=use_resnet,
            use_locally_connected=use_locally_connected,
            use_sensor_mask=use_sensor_mask,
            sensor_mask_regularizer_scale=sensor_mask_regularizer_scale,
            use_time_mask=use_time_mask,
            time_mask_regularizer_scale=time_mask_regularizer_scale,
            independent_channels=independent_channels,
            batch_normalize=batch_normalize,
            layer_normalize=layer_normalize,
            l2_layer_normalize=l2_layer_normalize,
            n_projection_layers=n_projection_layers,
            variational=variational,
            n_train=n_train * X_train.shape[1]
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )
        optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.999)

    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    metadata = {
        'ix2lab': ix2lab,
        'subject_map': subject_map,
        'glove_pca': glove_pca
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    if inner_validation_split:
        inner_cv_ix_path = os.path.join(fold_path, 'inner_cv_ix.obj')
        if force_resample_cv or force_restart or not os.path.exists(inner_cv_ix_path):
            # Shuffle training data
            perm = np.random.permutation(np.arange(len(X_train)))
            train_inner_ix = perm[:n_train]
            val_inner_ix = perm[n_train:]
            inner_cv_ix = {
                'train': train_inner_ix,
                'val': val_inner_ix
            }
            with open(inner_cv_ix_path, 'wb') as f:
                pickle.dump(inner_cv_ix, f)
        else:
            with open(inner_cv_ix_path, 'rb') as f:
                inner_cv_ix = pickle.load(f)
            train_inner_ix = inner_cv_ix['train']
            val_inner_ix = inner_cv_ix['val']

        y_val_lab_ix_inner = y_train[val_inner_ix]
        y_val_lab_inner = y_train_lab[val_inner_ix]
        if use_glove:
            y_val_glove_inner = y_train_glove[val_inner_ix]
            y_val_inner = y_val_glove_inner
            uniq, ix = np.unique(y_val_lab_ix_inner, return_index=True)
            comparison_set = {}
            for k, val in zip(ix, uniq):
                comparison_set[val] = y_val_glove_inner[k]
        else:
            y_val_inner = y_val_lab_ix_inner
            y_val_glove_inner = None

        ds_train = RasterData(
            X_train[train_inner_ix],
            y=y_train[train_inner_ix],
            # sample_weights=m.layers[-1].get_sample_weights(),
            batch_size=dnn_batch_size,
            shuffle=True,
            contrastive_sampling=bool(contrastive_loss_weight)
        )
        ds_val = RasterData(
            X_train[val_inner_ix],
            y=y_val_inner,
            batch_size=dnn_batch_size,
            shuffle=False,
            contrastive_sampling=bool(contrastive_loss_weight)
        )
    else:
        ds_train = RasterData(
            X_train,
            y=y_train,
            batch_size=dnn_batch_size,
            shuffle=True,
            contrastive_sampling=bool(contrastive_loss_weight)
        )
        ds_val = None
        y_val_inner = None
        y_val_lab_inner = None
        y_val_lab_ix_inner = None

    results_dict = {
        'acc': None,
        'f1': None,
        'chance_acc': chance_score_acc,
        'chance_f1': chance_score_f1,
    }
    callbacks = [
        # tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, verbose=1),
        # tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=100, restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path
        ),
        tfa.callbacks.AverageModelCheckpoint(
            update_weights=False,
            filepath=ema_path
        ),
        tf.keras.callbacks.TensorBoard(tb_path),
    ]

    if inner_validation_split:
        callbacks.append(
            ModelEval(
                model_path=ema_path,
                eval_freq=save_freq,
                eval_data_generator=ds_val,
                ylab=y_val_lab_inner,
                use_glove=use_glove,
                ix2lab=ix2lab,
                comparison_set=comparison_set,
                results_dict=results_dict,
                results_path=results_path
            )
        )

    if use_glove:
        loss = 'mse'
        # loss = tf.keras.losses.CosineSimilarity()
        metrics = []
        if reg_scale or use_sensor_mask or use_time_mask or variational:
            metrics.append('mse')
        metrics.append(tf.keras.metrics.CosineSimilarity(name='sim'))
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = []
        if reg_scale or use_sensor_mask or use_time_mask or variational:
            metrics.append('ce')
        metrics.append('acc')

    if force_restart or not os.path.exists(model_path):
        m.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    m.summary()

    batches_per_epoch = len(ds_train)
    n_batch_completed = 0
    for var in m.optimizer.variables():
        if var.name.startswith('iter'):
            n_batch_completed = var.numpy()

    initial_epoch = math.floor(n_batch_completed / batches_per_epoch)

    m.fit(
        ds_train,
        epochs=n_dnn_epochs,
        initial_epoch=initial_epoch,
        shuffle=False,
        callbacks=callbacks,
        validation_data=ds_val
    )

    if inner_validation_split:
        eval_and_save(
            ema_path,
            ds_val,
            y_val_lab_inner,
            results_path,
            use_glove=use_glove,
            ix2lab=ix2lab,
            comparison_set=comparison_set,
            results_dict=results_dict
        )

    def print_fn(x):
        with open(summary_path, 'a') as f:
            f.write(x + '\n')

    m.summary(print_fn=print_fn)
