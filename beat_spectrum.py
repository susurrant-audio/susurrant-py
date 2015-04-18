#!/usr/bin/env python

from sklearn.metrics.pairwise import pairwise_distances
import scipy
from librosa.core import get_duration
import numpy as np
import h5py
from scipy.fftpack import dct, idct
from utils import (WINDOW_SIZE, HOP_SIZE, SAMPLE_RATE,
                   COEFS, TIMBRE_GROUP)


ONE_FRAME = get_duration(sr=SAMPLE_RATE, n_fft=WINDOW_SIZE,
                         hop_length=HOP_SIZE,
                         S=np.zeros((1, 2)))


def beat_similarity(vecs):
    return 1.0 - pairwise_distances(vecs, metric='cosine')


def normed_diags(B, max_lag):
    B_diags = np.asarray([B.trace(i) for i in range(max_lag)])
    B_diags -= np.min(B_diags)
    return B_diags / np.max(B_diags)


def fft_autocorrelate(S):
    B_fft = scipy.signal.fftconvolve(S, S[::-1, ::-1], mode='full')
    B_fft = B_fft[S.shape[0]:, S.shape[1]:]
    return B_fft


def norm_fft(B_fft, max_lag=200, center_means=True, zero_min=True):
    B_kl = B_fft[:, :max_lag]
    if center_means:
        B_kl = (B_kl.T - np.mean(B_kl, axis=1)).T
    if zero_min:
        B_kl = (B_kl.T - B_kl.min(axis=1)).T
    maxes = np.max(B_kl, axis=1)
    maxes[maxes < 1.0] = 1.0  # clip
    B_kl = (B_kl.T * 1./maxes).T
    return B_kl


def beat_spectrum_fft(mfccs, analysis_win=None, max_lag=200):
    S = beat_similarity(mfccs)
    if analysis_win is not None:
        S *= analysis_win
    B_fft = fft_autocorrelate(S)
    B_kl = norm_fft(B_fft, max_lag)
    # assert np.isclose(np.min(B_kl), 0.0)
    assert np.isclose(np.max(B_kl), 1.0)
    return B_kl


def beat_spectrum(S_frame, max_lag, window_size=None):
    if window_size is None:
        window_size = S_frame.shape[0] - max_lag
    B = np.zeros((max_lag, max_lag), dtype=np.float64)
    S_0 = S_frame[:window_size, :window_size]
    S_0 -= np.mean(S_0)
    S_0_sq = np.square(S_0).sum()
    for k in range(max_lag):
        for l in range(k+1):
            lagged = S_frame[k:k+window_size, l:l+window_size]
            lagged -= np.mean(lagged)
            norm = np.sqrt(S_0_sq * np.square(lagged).sum())
            b = np.multiply(S_0, lagged).sum() / norm
            B[k, l] = b
            B[l, k] = b
    return normed_diags(B, max_lag)


def beat_spectrum_trace(S_frame, max_lag, window_size=None):
    if window_size is None:
        window_size = S_frame.shape[0] - max_lag
    B = np.zeros(max_lag, dtype=np.float64)
    S_0 = S_frame[:window_size, :window_size]
    S_0 -= np.mean(S_0)
    S_0_sq = np.square(S_0).sum()
    for l in range(max_lag):
        lagged = S_frame[l:l+window_size, l:l+window_size]
        lagged -= np.mean(lagged)
        norm = 1./np.sqrt(S_0_sq * np.square(lagged).sum())
        b = np.multiply(S_0, lagged).sum() * norm
        B[l] = b
    B -= np.min(B)
    return B * 1./np.max(B)


def beat_diag(S_frame, max_lag, window_size=None):
    if window_size is None:
        window_size = S_frame.shape[0] - max_lag
    S_0 = S_frame[:window_size, :window_size]
    return normed_diags(S_0, max_lag)


def beat_windows(S, max_lag, window_size=1024, hop_size=512, mode='diag'):
    n = int(np.floor(1.0 * S.shape[0] / window_size))
    rem = window_size - (S.shape[0] % window_size)
    extra = rem + window_size + max_lag
    S_padded = np.pad(S, ((0, extra), (0, extra)),
                      mode='constant',
                      constant_values=((0.0, 0.0), (0.0, 0.0)))
    np.fill_diagonal(S_padded, 1.0)
    hops = range(0, window_size, hop_size)
    for i in xrange(n):
        k = i * window_size
        S_spectra = []
        for hop in hops:
            frame = S_padded[k+hop:k+window_size+hop+max_lag,
                             k+hop:k+window_size+hop+max_lag]
            if mode == 'diag':
                spectrum = beat_diag(frame, max_lag, window_size)
            elif mode == 'trace':
                spectrum = beat_spectrum_trace(frame, max_lag, window_size)
            else:
                spectrum = beat_spectrum(frame, max_lag, window_size)
            S_spectra.append(spectrum)
        yield S_spectra


def track_mfccs(i=1, track_cutoff=None, keep_power=True):
    with h5py.File('../tracks.h5', 'r') as f:
        track = f.keys()[i]
        print track
        mfccs = f[track][TIMBRE_GROUP].value
    if track_cutoff is not None:
        mfccs = mfccs[:track_cutoff]
    if mfccs.shape[1] == COEFS and not keep_power:
        mfccs = mfccs[:, 1:]
    return mfccs


def win2D(L, win_func=np.hanning):
    w1D = win_func(L)
    M = (L-1)/2
    xx = np.linspace(-M, M, L)
    [x, y] = np.meshgrid(xx, xx)
    r = np.sqrt(np.square(x) + np.square(y))
    w2D = np.zeros((L, L), dtype=np.float64)
    w2D[r <= M] = np.interp(r[r <= M], xx, w1D)
    return w2D


def windowing(X, window_size=1024, hop_size=512, extra=None):
    X = np.asarray(X)
    n = X.shape[0] // window_size
    rem = window_size - (X.shape[0] % window_size)
    extra = rem + window_size
    X = np.pad(X, ((0, extra), (0, 0)),
               mode='constant',
               constant_values=((0.0, 0.0), (0.0, 0.0)))

    hops = range(0, window_size, hop_size)
    for i in xrange(n):
        k = i * window_size
        for hop in hops:
            if extra is not None:
                frame = X[k+hop:k+hop+window_size+extra]
            else:
                frame = X[k+hop:k+hop+window_size]
                if frame.shape[0] != window_size:
                    print "Padded to {}; didn't work".format(X.shape)
            yield frame


def flatten_windows(frames):
    out = []
    for hop_results in frames:
        M = reduce(np.add, hop_results)
        M *= 1./len(hop_results)
        out.append(M)
    return np.asarray(out)


def beat_spectra_from_track(i=0, window_size=1024, hop_size=256, mode='diag'):
    return beat_spectra_from_mfccs(track_mfccs(i), window_size=window_size,
                                   hop_size=hop_size, mode=mode)


def beat_spectra_from_mfccs_old(mfccs, window_size=1024, hop_size=256,
                                mode='diag'):
    S = beat_similarity(mfccs)
    frames = beat_windows(S, max_lag=200,
                          window_size=window_size, hop_size=hop_size,
                          mode=mode)
    return flatten_windows(frames)


def beat_spectra_from_mfccs(mfccs, window_size=1024, hop_size=256,
                            mode='diag'):
    windows = windowing(mfccs, window_size=window_size, hop_size=hop_size,
                        extra=200)
    for mfcc_sample in windows:
        S = beat_similarity(mfcc_sample)
        if mode == 'diag':
            yield beat_diag(S, max_lag=200)
        elif mode == 'trace':
            yield beat_spectrum_trace(S, max_lag=200)


def beat_coefs_from_mfccs(mfccs, window_size=1024, hop_size=256,
                          mode='diag'):
    spectra = beat_spectra_from_mfccs(mfccs, window_size=window_size,
                                      hop_size=hop_size, mode=mode)
    return beat_coefs(np.asarray([x for x in spectra]))


def beat_ffts_from_mfccs(mfccs, window_size=1024, hop_size=256,
                         mode='diag'):
    spectra = beat_spectra_from_mfccs(mfccs, window_size=window_size,
                                      hop_size=hop_size, mode=mode)
    return beat_fft(np.asarray([x for x in spectra]))[:, 1:25]


def beat_dcts_from_mfccs(mfccs, window_size=1024, hop_size=256,
                         mode='diag'):
    spectra = beat_spectra_from_mfccs(mfccs, window_size=window_size,
                                      hop_size=hop_size, mode=mode)
    return spectra_to_dct(spectra)


def pad_to(arr, width, front=1):
    arr = np.asarray(arr)
    return np.pad(arr, ((front, width - arr.size - front)),
                  mode='constant',
                  constant_values=(0.0, 0.0))


def spectra_to_dct(spectra):
    return np.asarray([dct(spec, norm='ortho') for spec in spectra])[:, 1:25]


def dct_to_spectra(dcts):
    return np.asarray([idct(pad_to(x, 200)) for x in dcts])


def fft_beat_spectra(mfccs, max_lag=200, window_size=1024, hop_size=512,
                     win_type='hanning'):
    n = mfccs.shape[0] // window_size
    B = np.zeros(((n + 1) * window_size, max_lag), dtype=np.float64)
    windows = windowing(mfccs, window_size=window_size, hop_size=hop_size)
    analysis_win = None
    if win_type == 'hanning':
        analysis_win = win2D(window_size, win_func=np.hanning)
    elif win_type == 'bartlett':
        analysis_win = win2D(window_size, win_func=np.bartlett)
    out_win = np.hanning(window_size - 1) * (hop_size * 2.0/window_size)
    for i, mfcc_sample in enumerate(windows):
        k = i * hop_size
        B_kl = beat_spectrum_fft(mfcc_sample, analysis_win)
        B_kl = (B_kl.T * out_win).T
        B[k:k+window_size-1, :] += B_kl
    return B


def fft_beat_spectra_avgs(mfccs, max_lag=200, window_size=1024, hop_size=512,
                          win_type='hanning'):
    n = mfccs.shape[0] // window_size
    B = np.zeros((n + 1, max_lag), dtype=np.float64)
    windows = windowing(mfccs, window_size=window_size, hop_size=hop_size)
    analysis_win = None
    if win_type == 'hanning':
        analysis_win = win2D(window_size, win_func=np.hanning)
    elif win_type == 'bartlett':
        analysis_win = win2D(window_size, win_func=np.bartlett)
    scale_out = hop_size * 2.0/window_size
    for i, mfcc_sample in enumerate(windows):
        win_k = i // (window_size / hop_size)
        B_kl = beat_spectrum_fft(mfcc_sample, analysis_win)
        B_l = B_kl.sum(axis=0)
        B_l *= 1./np.max(B_l)
        B_l *= scale_out
        B[win_k, :] += B_l
    return B


def fft_beat_spectra_from_track(i=2, max_lag=200,
                                window_size=1024, hop_size=512,
                                win_type=None, f_type='avgs'):
    if f_type == 'avgs':
        f = fft_beat_spectra_avgs
    else:
        f = fft_beat_spectra
    return f(track_mfccs(i), max_lag=max_lag,
             window_size=window_size,
             hop_size=hop_size, win_type=win_type)


def beat_fft(spectra, skip=1, skip_end=1):
    return np.asarray([np.log(np.absolute(np.fft.rfft(x)))
                       for x in spectra[skip:-skip_end]])


def beat_coefs_from_fft(f):
    dcts = scipy.fftpack.dct(f, norm='ortho')[:, 1:13]
    return dcts


def beat_coefs(spectra):
    return beat_coefs_from_fft(beat_fft(spectra)[:, 1:25])


def beat_spectrum_from_dct(dct_ex):
    return dct_to_spectra([dct_ex])


if __name__ == '__main__':
    print(fft_beat_spectra_from_track())
