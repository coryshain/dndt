import os
import pickle
import re
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import argparse
from matplotlib import pyplot as plt

fold_matcher = re.compile('i(\d+)_f(\d+)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run cross-validated decoding.
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to config file(s) containing decoding settings.')
    argparser.add_argument('-e', '--eval', default='', help='Which evaluation to plot. Defaults to plotting the cross-validated main objective.')
    args = argparser.parse_args()

    config_paths = args.config_paths
    eval = args.eval
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=Loader)

        confint = config.get('confint', 95)
        onset = config.get('onset', 0.5)
        downsample_by = config.get('downsample_by', 10)
        outdir = os.path.normpath(config.get('outdir', './results'))

        results = {
            'acc': [],
            'f1': []
        }
        chance = {
            'acc': [],
            'f1': [],
        }
        for i, j in [fold_matcher.match(x).groups() for x in os.listdir(outdir) if fold_matcher.match(x)]:
            folddir = os.path.join(outdir, 'i%s_f%s' % (i, j))
            if eval:
                filename = 'results_eval%s.obj' % eval
            else:
                filename = 'results.obj'
            respath = os.path.join(folddir, filename)
            if os.path.exists(respath):
                with open(respath, 'rb') as f:
                    _results = pickle.load(f)
                results['acc'].append(_results['acc'])
                results['f1'].append(_results['f1'])
                chance['acc'].append(_results['chance_acc'])
                chance['f1'].append(_results['chance_f1'])

        if len(results['acc']):
            results['acc'] = np.stack(results['acc'], axis=0)
            results['f1'] = np.stack(results['f1'], axis=0)
            chance['acc'] = np.mean(chance['acc'])
            chance['f1'] = np.mean(chance['f1'])

            for score in results:
                _results = results[score] * 100  # Place scores on 0-100 for readability
                _chance = chance[score] * 100  # Place scores on 0-100 for readability
                ntime = _results.shape[1]

                mean = _results.mean(axis=0)

                a = 100 - confint

                lb, ub = np.percentile(_results, (a / 2, 100 - (a / 2)), axis=0)
                x = np.linspace(0 - onset, ntime / 1000 * downsample_by - onset, ntime)

                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.xlim((x.min(), x.max()))
                plt.axvline(0, color='k')
                if score in ('acc', 'f1'):
                    plt.axhline(_chance, color='r')
                else:
                    raise ValueError('Unrecognized scoring function: %s' % score)
                plt.fill_between(x, lb, ub, alpha=0.2)
                plt.plot(x, mean)
                plt.xlabel('Time (s)')
                if score == 'acc':
                    plt.ylabel('% Correct')
                elif score == 'f1':
                    plt.ylabel('F1')
                else:
                    raise ValueError('Unrecognized scoring function: %s' % score)
                plt.tight_layout()

                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                if eval:
                    eval_str = '_eval%s' % eval
                else:
                    eval_str = ''
                plt.savefig(os.path.join(outdir, 'perf_plot%s_%s.png' % (eval_str, score)))

                plt.close('all')
