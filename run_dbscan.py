#!/usr/bin/env python
import sys
from cli import run_app

ELKI_JAR = '../susurrant-utils/lib/elki-bundle-0.6.5-20141030.jar'


def run_dbscan(in_file, out_dir):
    elki_opts = {
        '-algorithm': 'clustering.DBSCAN',
        '-dbc.in': in_file,
        '-dbscan.epsilon': 0.02,
        '-dbscan.minpts': 10,
        '-norm': 'AttributeWiseMinMaxNormalization',
        '-normUndo': None,
        '-time': None,
        '-out': out_dir
    }

    run_app('java', ['-jar', ELKI_JAR], elki_opts)

if __name__ == '__main__':
    run_dbscan(*sys.argv[1:])
