#!/usr/bin/env python

from annoy import AnnoyIndex
import numpy as np
import h5py
from utils import cached
from progressbar import ProgressBar

TREE_FILE = 'kmeans/5000.tree'

progress = ProgressBar()

features = 12
tree = AnnoyIndex(features, metric='euclidean')
tree.load(TREE_FILE)


def nn(x):
    return tree.get_nns_by_vector(x.tolist(), 1)[0]


@cached(fname='cached/annoy_50.npy')
def nns():
    with h5py.File('mfccs.h5', 'r') as f:
        dset = f['X']
        value = dset.value
    # assignments = np.apply_along_axis(nn, 1, dset)
    assignments = np.asarray([nn(x) for x in progress(value)])
    return assignments

assignments = nns()
actual = np.load('kmeans/assignments_5000.npy')

same = np.equal(assignments, actual)
print(float(same[same].size) / assignments.size)
