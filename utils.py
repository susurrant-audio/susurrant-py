#!/usr/bin/env python

from functools import wraps
import numpy as np
import os
import sys
import hashlib
import scipy
import librosa
from essentia.standard import *
from gammatone.fftweight import fft_weights
from constants import *

CACHE_DIR = '../cached'


def cached(fname=None, load_func=np.load, save_func=np.save):
    my_fname = fname

    def inner(f):
        fname = my_fname
        if fname is None:
            hash_str = hashlib.md5(f.__name__).hexdigest()
            fname = os.path.join(CACHE_DIR, hash_str)

        @wraps(f)
        def wrapper(*args, **kwargs):
            if os.path.exists(fname):
                return load_func(fname)
            else:
                res = f(*args, **kwargs)
                save_func(fname, res)
                return res
        return wrapper
    return inner


LOG10 = np.log(10)
LOG10_OVER_20 = LOG10 / 20.0

mel_fb = librosa.filters.mel(SAMPLE_RATE, WINDOW_SIZE, n_mels=BANDS,
                             fmin=LOW_FREQ,
                             fmax=HIGH_FREQ)
inv_mel_fb = np.linalg.pinv(mel_fb)


def inv_mel(warp_spectrogram):
    return np.dot(inv_mel_fb, warp_spectrogram)


erb_weights, gain = fft_weights(WINDOW_SIZE, SAMPLE_RATE,
                                COEFS,
                                1,
                                LOW_FREQ,
                                HIGH_FREQ,
                                WINDOW_SIZE/2 + 1)

inv_erb_weights = np.linalg.pinv(erb_weights)


def inv_erb(x):
    return np.dot(inv_erb_weights, x)


def ceps_to_stft(inv, x, has_power):
    if not has_power:
        val = 0.0
        x = np.concatenate(([val], x))
    x = x.astype(np.float64)
    log_spectrum = scipy.fftpack.idct(x, norm='ortho')
    warp_spectrogram = np.exp(log_spectrum * LOG10_OVER_20)
    pow_spectrum = inv(warp_spectrogram)
    energy_spectrum = np.sqrt(np.abs(pow_spectrum))
    return energy_spectrum


def mfcc_to_stft(x, has_power):
    return ceps_to_stft(inv_mel, x, has_power)


def gfcc_to_stft(x, has_power):
    return ceps_to_stft(inv_erb, x, has_power)


def invceps(to_stft, xs, has_power, recover_phase):
    x = np.asarray([to_stft(t, has_power) for t in xs]).T
    window = np.ones(WINDOW_SIZE) if HOP_FRAC == 2 else None
    if recover_phase:
        y = np.random.rand(WINDOW_SIZE * (x.shape[1]/2))
        x_hat = np.copy(x)
        for i in xrange(ISTFTM_ITER):
            D = librosa.core.stft(y, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
                                  window=scipy.signal.hann(WINDOW_SIZE))
            S, phase = librosa.core.magphase(D)
            if x_hat.shape[1] != phase.shape[1]:
                x_hat = np.pad(x_hat, ((0, 0), (0, 1)),
                               mode='constant',
                               constant_values=((0.0, 0.0), (0.0, 0.0)))
            y = librosa.core.istft(x_hat * phase, hop_length=HOP_SIZE,
                                   window=window)
    else:
        y = librosa.core.istft(x, hop_length=HOP_SIZE,
                               window=window,
                               # center=False,
                               dtype=np.float64)
    return y / np.max(np.abs(y))


def reconstruct_phase(X, y_len=None, window_size=WINDOW_SIZE,
                      hop_size=HOP_SIZE, iter_n=ISTFTM_ITER):
    hop_frac = window_size / hop_size
    if y_len is None:
        y_len = window_size * (X.shape[1]/2)
    window = np.ones(window_size) if hop_frac == 2 else None
    y = np.random.rand(y_len)
    x_hat = np.copy(X)
    for i in range(iter_n):
        D = librosa.core.stft(y, n_fft=window_size, hop_length=hop_size,
                              window=scipy.signal.hann(window_size))
        S, phase = librosa.core.magphase(D)
        if x_hat.shape[1] != phase.shape[1]:
            x_hat = np.pad(x_hat, ((0, 0), (0, 1)),
                           mode='constant',
                           constant_values=((0.0, 0.0), (0.0, 0.0)))
        y = librosa.core.istft(x_hat * phase, hop_length=hop_size,
                               window=window)
    return y / np.max(np.abs(y))


def imfcc(xs, has_power=False, recover_phase=True):
    return invceps(mfcc_to_stft, xs, has_power, recover_phase)


def igfcc(xs, has_power=False, recover_phase=True):
    return invceps(gfcc_to_stft, xs, has_power, recover_phase)
