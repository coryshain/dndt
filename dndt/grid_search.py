import os
import shutil
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import itertools
import argparse


def compute_hypercube(yml):
    params = {}
    for name in yml:
        v = yml[name]
        name = str(name)
        if isinstance(v, list):
            out_cur = {name: v}
        elif isinstance(v, dict):
            out_cur = v
        else:
            raise ValueError('Value of unrecognized type %d. Must be ``list`` or ``dict``. Value:\n%s' %(type(v), v))

        params[''.join(name.split('_'))] = out_cur

    names = sorted(list(params.keys()))
    search = []
    for n in names:
        search_cur = []
        if 'names' in params[n]:
            val_names = params[n].pop('names')
        else:
            val_names = []
        for k in params[n]:
            for i in range(len(params[n][k])):
                v = params[n][k][i]
                if i >= len(val_names):
                    val_name = v
                    val_names.append(val_names)
                else:
                    val_name = val_names[i]
                if i >= len(search_cur):
                    search_cur.append([n, val_name, (k, v)])
                else:
                    search_cur[i].append((k, v))
        search.append(search_cur)

    out = itertools.product(*search)

    for x in out:
        new = []
        name = []
        for y in x:
            new += y[2:]
            name.append(''.join([str(n) for n in y[:2]]))
        new = dict(new)
        name = '_'.join(name)

        yield new, name


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates model (*.ini) files from source templates, using Cartesian project of parameter settings defined in a
        YAML file. The YAML file must define a dictionary in which keys contain the user-specified name of the search
        dimension and values contain dictionaries from hyperparameter keys to lists of values to search.
        Multiple dictionaries given under the same name will be interpreted as co-varying, rather than being searched
        in a grid. An optional reserved key ``names`` allows the user to specify the name of each level in the
        search over a dimension. If omitted, the values of the levels from one of the dictionaries will be chosen.
        As a shortcut, an entry in the YAML file can also consist of a single map from a DNNSEG hyperparameter
        key to a list of values, in which case the key will also be used as the name for the search dimension.
        
        For example, to search over encoder depths vs. covarying levels of state and boundary noise, the following
        YAML string would be used:
        
        n_layers:
          - 2
          - 3
          - 4
        size:
          names:
            - S
            - M
            - L
          n_layers:
            - 1
            - 2
            - 3
          n_units:
            - 32
            - 64
            - 128
        
        This will create a 3 x 3 search over number of encoder layers vs. "size", with layers and units treated as 
        covarying -- i.e. S = (1, 32), M = (2, 64), and L = (3, 128) for layers and units, respectively.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to template *.ini file(s).')
    argparser.add_argument('search_params_path', help='Path to YAML file defining search params.')

    args = argparser.parse_args()
    search_params_path = args.search_params_path

    with open(search_params_path, 'r') as f:
        search = yaml.load(f, Loader=Loader)
        search_name = os.path.basename(search_params_path[:-4])
    params = compute_hypercube(search)

    for path in args.paths:
        template_name = os.path.basename(path[:-4])
        yml_outdir = os.path.join(template_name, search_name)
        if not os.path.exists(yml_outdir):
            os.makedirs(yml_outdir)
        with open(path, 'r') as f:
            template = yaml.load(f, Loader=Loader)
        results_outdir = os.path.normpath(template.get('outdir', './results'))
        results_outdir = os.path.join(results_outdir, template_name, search_name)
        if not os.path.exists(results_outdir):
            os.makedirs(results_outdir)
        shutil.copy2(search_params_path, os.path.join(results_outdir, 'search.yml'))

        for p, n in params:
            output = template.copy()
            filename = '_'.join((template_name, search_name, n)) + '.yml'
            yml_path = os.path.join(yml_outdir, filename)
            results_path = os.path.join(results_outdir, n)
            output['outdir'] = results_path
            for key in p:
                output[key] = p[key]
            with open(yml_path, 'w') as f:
                yaml.dump(output, f)
