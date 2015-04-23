#!/usr/bin/env python

import os
import json
import sys
import numpy as np
from annoy import AnnoyIndex
from index_clusters import get_tree_items
from utils import gfcc_to_stft
from beat_spectrum import beat_spectrum_from_dct
from constants import valid_data_types, FEATURES_N

VOCAB_DIR = '../vocab/'
DATA_DIR = '../susurrant_elm/data/'

inversions = {'gfccs': lambda x: gfcc_to_stft(x, False),
              'beat_coefs': beat_spectrum_from_dct,
              'chroma': lambda x: np.asarray(x)
              }


def get_vectors(tree_file, features):
    a = AnnoyIndex(features, metric='euclidean')
    a.load(tree_file)
    return get_tree_items(a)


def invert(inv_f, vectors):
    return [inv_f(np.asarray(vec)).tolist() for vec in vectors]


def vocab_vectors_by_dtype():
    vocabs = {}
    for dtype in valid_data_types:
        tree_file = os.path.join(VOCAB_DIR, 'train',
                                 'clusters_' + dtype + '.tree')
        vectors = get_vectors(tree_file, features=FEATURES_N[dtype])
        vocabs[dtype] = vectors
    return vocabs


def to_tokens(vectors_by_dtype):
    vocab = {}
    for dtype in valid_data_types:
        for i, vec in enumerate(vectors_by_dtype[dtype]):
            vocab[dtype + str(i)] = vec

    return vocab


def output_vectors(vocab_file=os.path.join(DATA_DIR, 'vocab.json')):
    vocab = to_tokens(vocab_vectors_by_dtype())
    with open(vocab_file, 'wb') as out:
        json.dump(vocab, out)


def invert_all(vocab_file=os.path.join(DATA_DIR, 'inverses.json')):
    vocabs = vocab_vectors_by_dtype()
    inverse_vocabs = {k: invert(inversions[k], v)
                      for k, v in vocabs.iteritems()
                      }
    inverse_vocab = to_tokens(inverse_vocabs)
    with open(vocab_file, 'wb') as out:
        json.dump(inverse_vocab, out)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'vocab':
        output_vectors(sys.argv[2])
    elif mode == 'invert':
        invert_all(sys.argv[2])
