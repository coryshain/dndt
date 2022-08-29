import os
import shutil
import argparse
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=%d:00:00
#SBATCH --mem=%dgb
#SBATCH --ntasks=%d
"""


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run decoder models specified in one or more config files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to CDR config file(s).')
    argparser.add_argument('-j', '--jobtype', default='fit', help='Type of job, one of "fit", "eval".')
    argparser.add_argument('-t', '--time', type=int, default=48, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n_cores', type=int, default=4, help='Number of cores to request')
    argparser.add_argument('-g', '--use_gpu', action='store_true', help='Whether to request a GPU node')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--slurm_partition', default=None, help='Value for SLURM --partition setting, if applicable')
    argparser.add_argument('-e', '--exclude', nargs='+', help='Nodes to exclude')
    argparser.add_argument('-s', '--singularity_path', default='', help='Path to singularity image to invoke before running')
    argparser.add_argument('-o', '--outdir', default='./', help='Directory in which to place generated batch scripts.')
    args = argparser.parse_args()

    paths = args.paths
    jobtype = args.jobtype
    time = args.time
    n_cores = args.n_cores
    use_gpu = args.use_gpu
    memory = args.memory
    slurm_partition = args.slurm_partition
    if args.exclude:
        exclude = ','.join(args.exclude)
    else:
        exclude = []
    singularity_path = args.singularity_path
    outdir = os.path.normpath(args.outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for path in [x for x in paths if x.endswith('.yml')]:
        with open(path, 'r') as f:
            c = yaml.load(f, Loader=Loader)

        niter = c['niter']
        nfolds = c['nfolds']
        model_outdir = os.path.normpath(c['outdir'])
        if not os.path.exists(model_outdir):
            os.makedirs(model_outdir)

        for iteration in range(niter):
            for fold in range(nfolds):
                job_name = os.path.basename(path)[:-4] + '_i%d_f%d' % (iteration+1, fold+1)
                filename = os.path.join(outdir, job_name + '.pbs')
                with open(filename, 'w') as f:
                    f.write(base % (job_name, job_name, time, memory, n_cores))
                    if use_gpu:
                        f.write('#SBATCH --gres=gpu:1\n')
                    if slurm_partition:
                        f.write('#SBATCH --partition=%s\n' % slurm_partition)
                    if exclude:
                        f.write('#SBATCH --exclude=%s\n' % exclude)
                    wrapper = '%s'
                    if singularity_path:
                        if use_gpu:
                            wrapper = wrapper % ('singularity exec --nv %s bash -c "cd %s; %%s"\n' % (singularity_path, os.getcwd()))
                        else:
                            wrapper = wrapper % ('singularity exec %s bash -c "cd %s; %%s"\n' % (singularity_path, os.getcwd()))

                    job_str = wrapper % ('python3 -m meg_coref.%s %s %s %s' % (jobtype, path, iteration+1, fold+1))
                    f.write(job_str)

