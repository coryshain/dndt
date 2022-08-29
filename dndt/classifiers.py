import numpy as np
from scipy.stats import rankdata, zscore
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from .nn import *


class MaxCorrClassifier:
    def __init__(self, k_feats=None, dropout=None):
        self.classes_ = []
        self.W = None
        self.k_feats = k_feats
        self.feat_selector = None
        self.dropout = dropout

    @property
    def n_classes(self):
        return len(self.classes_)

    def fit(self, X, y):
        assert len(X) == len(y), 'X and y must have the same length. Saw %d and %d, respectively.' % (len(X), len(y))
        unique, indices = np.unique(y, return_inverse=True)

        self.classes_ = unique

        if self.k_feats:
            self.feat_selector = SelectKBest(score_func=f_classif, k=self.k_feats)
            X = self.feat_selector.fit_transform(X, y[:, 0])

        W = np.zeros((X.shape[1], self.n_classes))
        for i, lab in enumerate(unique):
            ix = np.where(indices == i)
            _w = X[ix].mean(axis=0)
            W[:, i] = _w
        W = zscore(W, axis=0)
        if self.dropout:
            M = np.random.random(W.shape) > self.dropout
            W = W * M
        self.W = W

        return self

    def classify(self, X, argmax=True, comparison_set=None):
        if self.feat_selector:
            X = self.feat_selector.transform(X)
        X = zscore(X, axis=1)
        out = np.dot(X, self.W)
        if argmax:
            ix = np.argmax(out, axis=1)
            out = self.classes_[ix]

        return out


class _LogisticRegression(LogisticRegression):
    def classify(self, X, argmax=True, comparison_set=None):
        if argmax:
            out = super(_LogisticRegression, self).predict(X)
        else:
            out = self.predict_log_proba(X)

        return out


class _Ridge(Ridge):
    def classify(self, X, argmax=False, comparison_set=None):
        assert comparison_set is not None, 'Classification using ridge regression requires a comparison set'
        glove_pred = zscore(super(_Ridge, self).predict(X), axis=1)

        classes = np.array(sorted(comparison_set.keys()))
        glove_targ = np.stack([comparison_set[x] for x in classes], axis=1)
        glove_targ = zscore(glove_targ, axis=1)

        out = np.dot(glove_pred, glove_targ)
        if argmax:
            ix = np.argmax(out, axis=1)
            out = classes[ix]

        return out


class _LinearSVC(LinearSVC):
    def classify(self, X, argmax=True, comparison_set=None):
        if argmax:
            out = super(_LinearSVC, self).predict(X)
        else:
            out = self.predict_log_proba(X)

        return out


class _SVC(SVC):
    def classify(self, X, argmax=True, comparison_set=None):
        if argmax:
            out = super(_SVC, self).predict(X)
        else:
            out = self.predict_log_proba(X)

        return out


@ignore_warnings(category=ConvergenceWarning)
def train(X, y, n_dnn_epochs=1000, dnn_batch_size=32, clstype='LogisticRegression', **kwargs):
    args = []
    fit_kwargs = {}
    if clstype == 'LogisticRegression':
        reg = _LogisticRegression
    elif clstype == 'SVM':
        reg = _SVC
    elif clstype == 'LinearSVM':
        reg = _LinearSVC
    elif clstype == 'Ridge':
        reg = _Ridge
    elif clstype == 'MaxCorrelation':
        reg = MaxCorrClassifier
    else:
        raise ValueError('Unrecognized classifier %s' % clstype)

    def reg_fn(reg, args, kwargs, X, y, fit_kwargs):
        reg = reg(*args, **kwargs)
        return reg.fit(X, y, **fit_kwargs)

    out = reg_fn(reg, args, kwargs, X, y, fit_kwargs)

    return out


def classify(reg, X, argmax=True, comparison_set=None):
    kwargs = {}

    def classify_fn(reg, X, argmax, comparison_set, kwargs):
        return reg.classify(X, argmax=argmax, comparison_set=comparison_set, **kwargs)

    out = classify_fn(reg, X, argmax, comparison_set, kwargs)

    return out
