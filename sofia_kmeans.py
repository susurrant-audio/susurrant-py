#!/usr/bin/env python
# coding: utf-8

import subprocess
import re
import time
import os
import h5py
import sys
import logging
from constants import TIMBRE_GROUP, RHYTHM_GROUP, CHROMA_GROUP


data_types = {
    TIMBRE_GROUP: 500,
    RHYTHM_GROUP: 500,
    CHROMA_GROUP: 24
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_FILE = '/Users/chrisjr/Development/susurrant_prep/vocab'


def cluster(training_file, out_file, k=None):
    data_type = (os.path.basename(training_file)
                 .replace('.h5', '')
                 .replace('_sampled', ''))

    if k is None:
        k = data_types[data_type]

    logger.info('starting with k={}'.format(k))
    start = time.time()

    with h5py.File(training_file, 'r') as f:
        d = f['X'].shape[1]
    prog = ['/Users/chrisjr/Development/sofia-ml/sofia-kmeans',
            '--k', str(k),
            '--init_type', 'optimized_kmeans_pp',
            '--opt_type', 'mini_batch_kmeans',
            '--mini_batch_size', '100',
            '--iterations', '500',
            '--objective_after_training',
            '--training_file_h5', training_file,
            '--dimensionality', str(d + 1),
            '--model_out', out_file]
    result = subprocess.check_output(prog)
    objective_re = re.search("Objective function value for training: (.+)",
                             result)
    logger.info('{} finished in {} secs'.format(k, time.time() - start))
    if objective_re is not None:
        value = float(objective_re.group(1))
        logger.info("Objective for {}: {}".format(k, value))
        return value


# THIS IS TOO SLOW, use annoy or other ANN library instead

def make_assignments(training_file, out_dir='kmeans', k=500):
    logger.info('starting with k={}'.format(k))
    start = time.time()
    base_dir = os.path.join('/Users/chrisjr/Development/susurrant_prep/vocab',
                            out_dir)
    in_file = os.path.join(base_dir, 'clusters_rand_{}.txt'.format(k))
    assign_file = os.path.join(base_dir, 'assignments_{}.txt'.format(k))
    prog = ['/Users/chrisjr/Development/sofia-ml/sofia-kmeans',
            '--model_in', in_file,
            '--test_file_h5', training_file,
            '--cluster_assignments_out', assign_file,
            '--objective_on_test']
    result = subprocess.check_output(prog)
    logger.info('{} finished in {} secs'.format(k, time.time() - start))
    objective_re = re.search("Objective function value for test: (.+)", result)
    if objective_re is not None:
        value = float(objective_re.group(1))
        logger.info("Objective for {}: {}".format(k, value))
        return value

if __name__ == '__main__':
    if len(sys.argv) == 3:
        [training_file, out_file] = sys.argv[1:]
        k = None
    elif len(sys.argv) == 4:
        [training_file, out_file, k] = sys.argv[1:]
    res = cluster(training_file, out_file, k)
    print 'Objective value: {} after training'.format(res)
