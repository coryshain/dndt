import sys
import os
import re
import pickle
import numpy as np
from scipy.io import loadmat
import mne


def stderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()


def partition_cv(data, labels, nfolds, eval_filter_mask=None):
    label_keys = labels[:,0]
    labels_out = {}

    if eval_filter_mask is not None:
        supp_mask = ~eval_filter_mask
        data_supp = data[supp_mask]
        label_keys_supp = label_keys[supp_mask]
        labels_supp = labels[supp_mask]
        data = data[eval_filter_mask]
        label_keys = label_keys[eval_filter_mask]
        labels = labels[eval_filter_mask]

        p = np.random.permutation(np.arange(len(label_keys_supp)))
        data_supp = data_supp[p]
        label_keys_supp = label_keys_supp[p]
        labels_supp = labels_supp[p]
        unique, indices, counts = np.unique(label_keys_supp, return_inverse=True, return_counts=True)
        supp_out = {}

        for i, lab in enumerate(unique):
            ix = np.where(indices == i)
            _data_supp = data_supp[ix]
            labels_out[lab] = labels_supp[ix][0] # All labels in group identical, take first
            supp_out[lab] = _data_supp
    else:
        supp_out = {}

    p = np.random.permutation(np.arange(len(label_keys)))
    data = data[p]
    label_keys = label_keys[p]
    labels = labels[p]
    unique, indices, counts = np.unique(label_keys, return_inverse=True, return_counts=True)
    data_out = [{} for _ in range(nfolds)]

    for i, lab in enumerate(unique):
        ix = np.where(indices == i)
        _data = np.array_split(data[ix], nfolds, axis=0)
        labels_out[lab] = labels[ix][0] # All labels in group identical, take first
        for j, __data in enumerate(_data):
            data_out[j][lab] = __data

    return data_out, labels_out, supp_out


def compute_filter_mask(y, filters):
    for x in y:
        sel = np.ones(len(y[x]), dtype=bool)
        break
    for filter in filters:
        vals = filters[filter]
        if vals == 'contentwords':
            from nltk.corpus import stopwords
            try:
                stops = stopwords.words('english')
            except LookupError:
                import nltk
                nltk.download('stopwords')
                stops = stopwords.words('english')
            words = [w.lower() for w in y[filter]]
            stops = list(stops)
            sel &= ~np.isin(words, stops)
        elif isinstance(vals, list):
            sel &= np.isin(y[filter], vals)
        elif vals.startswith('<='):
            sel &= y[filter] <= float(vals[2:].strip())
        elif vals.startswith('<'):
            sel &= y[filter] < float(vals[1:].strip())
        elif vals.startswith('>='):
            sel &= y[filter] >= float(vals[2:].strip())
        elif vals.startswith('>'):
            sel &= y[filter] > float(vals[1:].strip())
        elif vals.startswith('=='):
            sel &= y[filter] == float(vals[2:].strip())
        elif vals.startswith('='):
            sel &= y[filter] == float(vals[1:].strip())
        elif vals.startswith('!='):
            sel &= y[filter] != vals[2:].strip()
        else:
            raise ValueError('Unrecognized filter: %s' % filter)

    return sel


def normalize(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / n


def compile_data(
        dirpath,
        downsample_by=1,
        powerband=None,
        force_reprocess=False,
):
    dirpath = os.path.normpath(dirpath)
    filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
    cache_path = os.path.join(dirpath, filename)

    if force_reprocess or not os.path.exists(cache_path):
        raster_data = []
        _labels = None
        raster_paths = [os.path.join(dirpath, x) for x in os.listdir(dirpath) if x.endswith('mat')]
        meta = {}
        for i in range(len(raster_paths)):
            raster_path = raster_paths[i]
            stderr('\r  Sensor %d/%d' % (i + 1, len(raster_paths)))
            raster = loadmat(raster_path, simplify_cells=True)
            _data = raster['raster_data']
            if powerband:
                meta['powerband'] = powerband
                l, u = powerband
                _data = mne.io.RawArray(
                    _data,
                    mne.create_info([str(x) for x in range(len(_data))], 1000, 'grad'),
                    verbose=40
                )
                _data = _data.filter(l, u, verbose=40) \
                    .apply_hilbert(envelope=True, verbose=40) \
                    .get_data()
            if downsample_by > 1:
                meta['downsample_by'] = downsample_by
                b = _data.shape[0]
                t = _data.shape[1]
                trim = t % downsample_by
                _t = t // downsample_by
                _data = _data[:, trim:].reshape((b, _t, downsample_by)).mean(axis=-1)
                # num = _data.shape[1] // downsample_by
                # _data = resample(_data, num, axis=1)
            raster_data.append(_data)
            if _labels is None:
                _labels = raster['raster_labels']

        _data = np.stack(raster_data, axis=1)

        data_src = {
            'data': _data,
            'labels': _labels,
            'meta': meta
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data_src, f)

        stderr('\n')


def compile_cv_ix(data, outdir, niter=1, nfolds=5, force_resample=False):
    outdir = os.path.normpath(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(niter):
        p = np.arange(len(data))
        p = np.random.permutation(p)
        p = np.array_split(p, nfolds, axis=0)
        for j in range(nfolds):
            train_ix = np.concatenate([_p for k, _p in enumerate(p) if k != j], axis=0)
            val_ix = p[j]

            train_ix_filename = 'train_ix_i%d_f%d.obj' % (i + 1, j + 1)
            train_ix_path = os.path.join(outdir, train_ix_filename)
            if force_resample or not os.path.exists(train_ix_path):
                with open(train_ix_path, 'wb') as f:
                    pickle.dump(train_ix, f)
                val_ix_filename = 'val_ix_i%d_f%d.obj' % (i + 1, j + 1)
                val_ix_path = os.path.join(outdir, val_ix_filename)
                with open(val_ix_path, 'wb') as f:
                    pickle.dump(val_ix, f)


def check_cv_ix(outdir):
    train_ix = {}
    val_ix = {}
    outdir = os.path.normpath(outdir)
    train_ix_paths = [os.path.join(outdir, x) for x in os.listdir(outdir) if 'train_ix' in x]
    val_ix_paths = [os.path.join(outdir, x) for x in os.listdir(outdir) if 'val_ix' in x]
    for path in train_ix_paths:
        iteration, fold = re.search('train_ix_i(\d+)_f(\d+).obj', path).groups()
        with open(path, 'rb') as f:
            train_ix[(iteration, fold)] = pickle.load(f)
    for path in val_ix_paths:
        iteration, fold = re.search('val_ix_i(\d+)_f(\d+).obj', path).groups()
        with open(path, 'rb') as f:
            val_ix[(iteration, fold)] = pickle.load(f)

    assert len(train_ix) == len(val_ix), \
        'Found different numbers of folds for train (%d) and validation (%d)' % (len(train_ix), len(val_ix))

    for x in train_ix:
        iteration, fold = x
        _train_ix = train_ix[x]
        _val_ix = val_ix[x]
        _ix = np.sort(np.concatenate([_train_ix, _val_ix], axis=0))
        if not np.all(np.equal(_ix, np.arange(len(_ix)))):
            print('CV check failed. Saved indices do not reconstruct the input data.')
            print('Iteration %s, Fold %s' % (iteration, fold))
            print(_ix)
            print(len(np.unique(_ix)))
            print(np.arange(len(_ix)))
            raise ValueError('CV check not passed.')
