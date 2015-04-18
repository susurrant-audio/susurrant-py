#!/usr/bin/env python

from pyflann import FLANN
import numpy as np
import h5py
from utils import cached

CLUSTER_FILE = 'kmeans/clusters_rand_5000.txt'
ALL_MFCCS = 'mfccs.h5'
FLANN_RESULTS = 'assignments_5000_flann.npy'

clusters = np.loadtxt(CLUSTER_FILE, dtype='<f4')[:, 1:]

flann = FLANN()


@cached(fname=FLANN_RESULTS)
def nns():
    mk_index()
    with h5py.File(ALL_MFCCS, 'r') as f:
        dset = f['X']
        result, dists = flann.nn_index(dset.value)
        return result


@cached(save_func=lambda x, _: flann.save_index(x),
        load_func=lambda x: flann.load_index(x, clusters))
def mk_index():
    flann.build_index(clusters, target_precision=0.99)

result = nns()
actual = np.load('kmeans/assignments_5000.npy')

same = np.equal(result, actual)
print(float(same[same].size) / result.size)
