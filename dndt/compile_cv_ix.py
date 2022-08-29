import os
import shutil
import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
import argparse

from .util import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run cross-validated decoding.
    ''')
    argparser.add_argument('config', help='Path to config file containing decoding settings.')
    argparser.add_argument('-f', '--force_resample', action='store_true', help='Force resampling of CV partition indices, even if samples already exist. Otherwise, partition indices will not be overwritten.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    force_resample = args.force_resample

    paths = config['paths']
    powerband = config.get('powerband', None)
    downsample_by = config.get('downsample_by', 1)
    nfolds = config.get('nfolds', 5)
    assert nfolds > 1, "nfolds must be >= 2."
    niter = config.get('niter', 10)
    separate_subjects = config.get('separate_subjects', True)
    combine_subjects = config.get('combine_subjects', True)
    outdir = config.get('outdir', './results')
    outdir = os.path.normpath(outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.normpath(os.path.realpath(config_path)) == os.path.normpath(os.path.realpath(outdir + '/config.ini')):
        shutil.copy2(config_path, outdir + '/config.ini')

    for dirpath in paths:
        dirpath = os.path.normpath(dirpath)
        stderr('Loading %s...\n' % dirpath)
        filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
        cache_path = os.path.join(dirpath, filename)
        with open(cache_path, 'rb') as f:
            data_src = pickle.load(f)
        _data = data_src['data']

        compile_cv_ix(_data, dirpath, niter=niter, nfolds=nfolds, force_resample=force_resample)
