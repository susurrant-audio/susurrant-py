#!/usr/bin/env python

import os
import json
import sys
import numpy as np
from annoy import AnnoyIndex
from index_clusters import get_tree_items
from utils import gfcc_to_stft, reconstruct_phase
from beat_spectrum import beat_spectrum_from_dct
from constants import *
from sklearn.preprocessing import scale, normalize
import librosa
import scipy.signal
from scikits.audiolab import wavwrite, play
from scipy.signal import square, sawtooth

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


def vocab_vectors_by_dtype(do_scale=False):
    vocabs = {}
    for dtype in valid_data_types:
        tree_file = os.path.join(VOCAB_DIR, 'train',
                                 'clusters_' + dtype + '.tree')
        vectors = get_vectors(tree_file, features=FEATURES_N[dtype])
        vocabs[dtype] = scale(vectors) if do_scale else vectors
    return vocabs


def to_tokens(vectors_by_dtype):
    vocab = {}
    for dtype in valid_data_types:
        for i, vec in enumerate(vectors_by_dtype[dtype]):
            vocab[dtype + str(i)] = list(vec)

    return vocab


audio_time = 0.5
audio_len = int(audio_time * 44100)

def midi_freq(n):
    return 440.0 * 2**((n-69.0)/12)

def saw(i, vol):
    t = np.linspace(0, audio_time, audio_len)
    freq = midi_freq(60+i)
#    print audio_time, audio_len
    return sawtooth(2 * np.pi * freq * t, width = 0.5) * vol #  np.power(vol, 1.5)

def gen_chroma(vec):
    vec = normalize(vec, axis=1, norm='l1')
    chroma_sounds = np.asarray([saw(i, vol) for i, vol in enumerate(vec[0])])
    return chroma_sounds.sum(axis=0)

def gen_timbre(vec):
    D = np.tile(vec, (50, 1)).T
    y = reconstruct_phase(D, iter_n=50)
    return y.astype(np.float64)


def play_audio():
    for dtype, inverted_vocab in get_inverse_vocabs().iteritems():
        if dtype != TIMBRE_GROUP:
            continue
        for i, vec in enumerate(inverted_vocab):
            play(gen_timbre(vec))


def to_audio(wav_file, index_file):
    audio_snippets = {}
    for dtype, inverted_vocab in get_inverse_vocabs().iteritems():
        if dtype == RHYTHM_GROUP:
            continue
        elif dtype == CHROMA_GROUP:
            inverted_vocab = scale(inverted_vocab)
            inverted_vocab[inverted_vocab < 0.0] = 0.0
        for i, vec in enumerate(inverted_vocab):
            token = dtype + str(i)
            if dtype == CHROMA_GROUP:
                audio_snippets[token] = gen_chroma(vec)
            elif dtype == TIMBRE_GROUP:
                audio_snippets[token] = gen_timbre(vec)
    keys = audio_snippets.keys()
    index = {}
    for i, key in enumerate(keys):
        t = i * audio_time
        end_t = (i+1) * audio_time
        index[key] = [t, end_t]
    with open(index_file, 'wb') as f:
        json.dump(index, f)
    wavwrite(np.hstack([audio_snippets[key] for key in keys]), wav_file, fs=44100)


def get_inverse_vocabs():
    vocabs = vocab_vectors_by_dtype()
    return {k: invert(inversions[k], v)
            for k, v in vocabs.iteritems()
            }


def output_vectors(vocab_file=os.path.join(DATA_DIR, 'vocab.json')):
    vocab = to_tokens(vocab_vectors_by_dtype(do_scale=True))
    with open(vocab_file, 'wb') as out:
        json.dump(vocab, out)


def invert_all(vocab_file=os.path.join(DATA_DIR, 'inverses.json')):
    inverse_vocabs = get_inverse_vocabs()
    inverse_vocab = to_tokens(inverse_vocabs)
    with open(vocab_file, 'wb') as out:
        json.dump(inverse_vocab, out)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'vocab':
        output_vectors(sys.argv[2])
    elif mode == 'invert':
        invert_all(sys.argv[2])
    elif mode == 'audio_file':
        to_audio(*sys.argv[2:])
    elif mode == 'audio':
        play_audio()
