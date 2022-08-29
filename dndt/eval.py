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
    Evaluate pretrained decoder
    ''')
    argparser.add_argument('config', help='Path to config file containing decoding settings.')
    argparser.add_argument('iteration', type=int, help='Which CV iteration to run (1-indexed).')
    argparser.add_argument('fold', type=int, help='Which CV fold to run (1-indexed).')
    argparser.add_argument('-c', '--cpu_only', action='store_true', help='Force CPU implementation if GPU available.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    nfolds = config.get('nfolds', 5)
    assert nfolds > 1, "nfolds must be >= 2."
    niter = config.get('niter', 10)
    powerband = config.get('powerband', None)
    downsample_by = config.get('downsample_by', 1)
    use_glove = config.get('use_glove', False)
    zscore_time = config.get('zscore_time', False)
    zscore_sensors = config.get('zscore_sensors', False)
    normalize_sensors = config.get('normalize_sensors', False)
    tanh_transform = config.get('tanh_transform', False)
    rank_transform_sensors = config.get('rank_transform_sensors', False)
    k_feats = config.get('k_feats', 0)
    k_pca_glove = config.get('k_pca_glove', 0)
    k_pca_drop = config.get('k_pca_drop', 0)
    layer_type = config.get('layer_type', 'cnn')
    learning_rate = config.get('learning_rate', 0.0001)
    n_units = config.get('n_units', 128)
    n_layers = config.get('n_layers', 1)
    kernel_width = config.get('kernel_width', 10)
    cnn_activation = config.get('cnn_activation', 'relu')
    reg_scale = config.get('reg_scale', None)
    sensor_filter_scale = config.get('sensor_filter_scale', None)
    dropout = config.get('dropout', None)
    input_dropout = config.get('input_dropout', None)
    temporal_dropout = config.get('temporal_dropout', None)
    use_resnet = config.get('use_resnet', False)
    use_locally_connected = config.get('use_locally_connected', False)
    independent_channels = config.get('independent_channels', False)
    batch_normalize = config.get('batch_normalize', False)
    layer_normalize = config.get('layer_normalize', False)
    l2_layer_normalize = config.get('l2_layer_normalize', False)
    n_projection_layers = config.get('n_projection_layers', 1)
    variational = config.get('variational', False)
    contrastive_loss_weight = config.get('contrastive_loss_weight', None)
    index_subjects = config.get('index_subjects', False)
    n_dnn_epochs = config.get('n_dnn_epochs', 1000)
    dnn_batch_size = config.get('dnn_batch_size', 32)
    inner_validation_split = config.get('inner_validation_split', None)
    score = config.get('score', 'acc')
    confint = config.get('confint', 95)
    onset = config.get('onset', 0.5)
    outdir = config.get('outdir', './results')

    iteration = args.iteration
    fold = args.fold

    stderr('Loading saved checkpoint...\n')
    fold_path = os.path.join(os.path.normpath(outdir), 'i%d_f%d' % (iteration, fold))
    model_path = os.path.join(fold_path, 'model_ema.h5')
    metadata_path = os.path.join(fold_path, 'metadata.obj')
    m = tf.keras.models.load_model(model_path)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        ix2lab_src = metadata['ix2lab']
        subject_map = metadata.get('subject_map', {})
        glove_pca = metadata.get('glove_pca', None)
    lab2ix_src = {ix2lab_src[x]: x for x in ix2lab_src}

    evaluations = config['evaluations']
    for eval_ix, evaluation in enumerate(evaluations):
        results_path = os.path.join(fold_path, 'results_eval%d.obj' % (eval_ix + 1))
        paths = evaluation['paths']
        label_field = evaluation['label_field']
        in_cv = evaluation['in_cv']
        filters = evaluation.get('filters', {})
        X = []
        y = []
        ntime = None
        n_subject = len(subject_map)
        for i, dirpath in enumerate(paths):
            dirpath = os.path.normpath(dirpath)
            stderr('Loading %s...\n' % dirpath)
            filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
            cache_path = os.path.join(dirpath, filename)
            with open(cache_path, 'rb') as f:
                data_src = pickle.load(f)
            _data = data_src['data']
            _labels = data_src['labels']
            meta = data_src['meta']

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
                if dirpath in subject_map:
                    six = subject_map[dirpath]
                    subject_ix[..., six] = 1
                _data = np.concatenate([_data, subject_ix], axis=-1)

            if in_cv:
                val_ix_path = os.path.join(dirpath, 'val_ix_i%d_f%d.obj' % (iteration, fold))
                with open(val_ix_path, 'rb') as f:
                    val_ix = pickle.load(f)
                _data = _data[val_ix]
                _labels = _labels[val_ix]

                _filter_mask = compute_filter_mask(_label_df.iloc[val_ix], filters)
                _data = _data[_filter_mask]
                _labels = _labels[_filter_mask]
            else:
                _filter_mask = compute_filter_mask(_label_df, filters)
                _data = _data[_filter_mask]
                _labels = _labels[_filter_mask]

            X.append(_data)
            y.append(_labels)

            if ntime is None:
                ntime = _data.shape[1]

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        labs, lab_ix, counts = np.unique(y[:, 0], return_index=True, return_counts=True)
        _nclass = len(labs)
        _maj_class = labs[counts.argmax()]
        _maj_class_pred = np.array([_maj_class])
        _baseline_preds = np.tile(_maj_class_pred, [len(y)])
        chance_score_acc = accuracy_score(y[:, 0], _baseline_preds)
        _probs = counts / counts.sum()
        _baseline_preds = np.random.multinomial(1, _probs, size=len(y))
        _baseline_preds = _baseline_preds.argmax(axis=1)
        _baseline_preds = labs[_baseline_preds]
        chance_score_f1 = f1_score(y[:, 0], _baseline_preds, average='macro')
        lab2ix = {_y: i for i, _y in enumerate(labs)}
        ix2lab = {lab2ix[_y]: _y for _y in lab2ix}
        y_lab = y[:, 0]
        if use_glove:
            if k_pca_glove:
                stderr('PCA-transforming GloVe components...\n')
                y = np.concatenate([y[:, :1], glove_pca.transform(y[:, 1:])], axis=1)
            y_glove = y[:, 1:].astype('float32')
            comparison_set = {}
            for k, val in zip(lab_ix, labs):
                comparison_set[val] = y_glove[k]
        else:
            y_glove = None
            comparison_set = sorted([lab2ix_src[x] for x in lab2ix])

        ds_eval = RasterData(X, batch_size=dnn_batch_size, shuffle=False)

        results_dict = {
            'acc': None,
            'f1': None,
            'chance_acc': chance_score_acc,
            'chance_f1': chance_score_f1,
        }

        eval_and_save(
            model_path,
            ds_eval,
            y_lab,
            results_path,
            use_glove=use_glove,
            ix2lab=ix2lab_src,
            comparison_set=comparison_set,
            results_dict=results_dict
        )
