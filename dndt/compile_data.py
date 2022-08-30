import os
import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import argparse
import mne

from .util import stderr, compile_data

MAX_N_RESAMP = 100


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Cache data for loading into decoder.
    ''')
    argparser.add_argument('config', help='Path to config file containing paths to MEG raster data and preprocessing instructions.')
    argparser.add_argument('-f', '--force_reprocess', action='store_true', help='Force data reprocessing, even if a data cache exists.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    force_reprocess = args.force_reprocess

    paths = config['paths']
    downsample_by = config.get('downsample_by', 1)
    powerband = config.get('powerband', None)
    outdir = config.get('outdir', './results')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data = []
    labels = []
    for dirpath in paths:
        stderr('Loading %s...\n' % dirpath)
        compile_data(
            dirpath,
            downsample_by=downsample_by,
            powerband=powerband,
            force_reprocess=force_reprocess
        )
