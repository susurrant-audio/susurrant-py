#!/usr/bin/env python

from annoy import AnnoyIndex
import numpy as np
import h5py
import sys


def create_index(cluster_file,
                 tree_file=None):
    if cluster_file.endswith('.h5'):
        with h5py.File(cluster_file, 'r') as f:
            clusters = f['centers'].value
    else:
        clusters = np.loadtxt(cluster_file)[:, 1:]

    if tree_file is None:
        tree_file = cluster_file.replace('.txt', '.tree')

    features = clusters.shape[1]
    tree = AnnoyIndex(features, metric='euclidean')

    for i, v in enumerate(clusters):
        tree.add_item(i, v.tolist())

    tree.build(features*2)
    tree.save(tree_file)
    return tree_file


if __name__ == '__main__':
    if len(sys.argv) == 2:
        create_index(sys.argv[1])
    elif len(sys.argv) == 3:
        create_index(*sys.argv[1:])
    else:
        print "Usage: index_clusters.py cluster_file index_file"
