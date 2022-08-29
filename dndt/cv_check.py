import argparse

from meg_coref.util import check_cv_ix

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Check whether CV splitting had any bugs. 
    ''')
    argparser.add_argument('path', help='Path to data directory containing CV split.')
    args = argparser.parse_args()

    path = args.path

    check_cv_ix(path)

