#!/usr/bin/env python
import sys
from cli import run_app

SUSURRANT_JAR = ('../susurrant-utils/target/scala-2.10/' +
                 'susurrant-utils-assembly-0.0.1.jar')


def run_dbscan(in_file, out_dir):
    elki_opts = {
        '-algorithm': 'clustering.DBSCAN',
        '-dbc': 'org.chrisjr.susurrantutils.Hdf5DatabaseConnection',
        '-h5.input': in_file,
        '-dbc.filter': '.'.join(['normalization',
                                 'columnwise',
                                 'AttributeWiseMinMaxNormalization']),
        '-db.index': 'tree.spatial.rstarvariants.rstar.RStarTreeFactory',
        '-spatial.bulkstrategy': 'SortTileRecursiveBulkSplit',
        '-dbscan.epsilon': 0.02,
        '-dbscan.minpts': 10,
        '-out': out_dir
    }

    run_app('java',
            ['-Xmx4g',
             '-cp',
             SUSURRANT_JAR,
             'de.lmu.ifi.dbs.elki.application.KDDCLIApplication'],
            elki_opts)

if __name__ == '__main__':
    run_dbscan(*sys.argv[1:])
