#!/usr/bin/env python

from annoy import AnnoyIndex
import numpy as np
import h5py
import os
import sys
from constants import TIMBRE_GROUP, valid_data_types, FEATURES_N
from progressbar import ProgressBar

BASE_DIR = '/Users/chrisjr/Development/susurrant_prep'


class FeatureNN:
    tree = None

    def __init__(self, features, tree_file):
        self.tree = AnnoyIndex(features, metric='euclidean')
        self.tree.load(str(tree_file))

    def nn(self, x):
        return self.tree.get_nns_by_vector(x.tolist(), 1)[0]


def get_anns(base_path=os.path.join(BASE_DIR, 'vocab', 'train')):
    anns = {}
    for data_type in valid_data_types:
        features = FEATURES_N[data_type]
        tree_file = os.path.join(base_path, 'clusters_' + data_type + '.tree')
        anns[data_type] = FeatureNN(features, tree_file)
    return anns


def tracks_to_assignments(track_file=os.path.join(BASE_DIR, 'segmented.h5'),
                          token_file=os.path.join(BASE_DIR, 'vocab',
                                                  'tokens.h5')):
    progress = ProgressBar()
    anns = get_anns()

    with h5py.File(track_file, 'r') as f:
        # check an examplar to make sure features_N is up to date
        ex = f.values()[0]
        for data_type in ex:
            features = ex[data_type].shape[1]
            if data_type == TIMBRE_GROUP:
                features -= 1
            assert features == FEATURES_N[data_type]
        with h5py.File(token_file) as g:
            for track in progress(f):
                grp = f[track]
                if track in g or set(grp.keys()) != valid_data_types:
                    continue
                out_grp = g.create_group(track)
                for t, tree in anns.iteritems():
                    values = grp[t].value
                    if t == TIMBRE_GROUP:
                        values = values[:, 1:]
                    assigned = [tree.nn(x) for x in values]
                    out_grp.create_dataset(t, data=np.asarray(assigned))


if __name__ == '__main__':
    tracks_to_assignments(*sys.argv[1:])
