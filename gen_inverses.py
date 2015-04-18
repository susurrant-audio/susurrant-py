#!/usr/bin/env python

import os
import json
import sys
import numpy as np
from annoy import AnnoyIndex
from utils import gfcc_to_stft
from beat_spectrum import beat_spectrum_from_dct
from constants import valid_data_types

VOCAB_DIR = '../vocab/'
DATA_DIR = '../susurrant_elm/data/'

inversions = {'gfccs': lambda x: gfcc_to_stft(x, False),
              'beat_coefs': beat_spectrum_from_dct,
              'chroma': lambda x: np.asarray(x)
              }


def get_vectors(tree_file, features):
    a = AnnoyIndex(features, metric='euclidean')
    a.load(tree_file)
    return [a.get_item_vector(i+1) for i in xrange(a.get_n_items())]


def invert(inv_f, tree_file, features):
    vectors = get_vectors(tree_file, features)
    return [inv_f(np.asarray(vec)).tolist() for vec in vectors]


def invert_all(vocab_file=os.path.join(DATA_DIR, 'vocab.json')):
    vocab = {}
    for dtype in valid_data_types:
        tree_file = os.path.join(VOCAB_DIR, 'train',
                                 'clusters_' + dtype + '.tree')
        vocab[dtype] = invert(inversions[dtype], tree_file,
                              features=(24 if dtype == 'beat_coefs' else 12))
    with open(vocab_file, 'wb') as out:
        json.dump(vocab, out)


if __name__ == '__main__':
    invert_all(sys.argv[1])
