#!/usr/bin/env python

import os
import h5py
import logging
import sys
from essentia.streaming import *
from ipy_progressbar import ProgressBar
from constants import (WINDOW_SIZE, HOP_SIZE, SAMPLE_RATE,
                       TIMBRE_GROUP, RHYTHM_GROUP, CHROMA_GROUP,
                       LOW_FREQ, HIGH_FREQ, BANDS, COEFS)
from beat_spectrum import beat_dcts_from_mfccs


def analyze_tracks(track_dir,
                   track_file='../tracks.h5'):
    files = [x for x in os.listdir(track_dir)
             if x.endswith('.mp3') or x.endswith('.wav')]

    w = Windowing(type='hann', size=WINDOW_SIZE)
    spectrum = Spectrum()
    options = {'sampleRate': SAMPLE_RATE,
               'numberBands': BANDS,
               'numberCoefficients': COEFS,
               'lowFrequencyBound': LOW_FREQ,
               'highFrequencyBound': HIGH_FREQ
               }
    gfcc = GFCC(**options)
    fcc_name = 'lowlevel.gfcc'

    framecutter = FrameCutter(frameSize=WINDOW_SIZE, hopSize=HOP_SIZE)
    peaks = SpectralPeaks(sampleRate=SAMPLE_RATE)
    hpcp = HPCP(sampleRate=SAMPLE_RATE)
    pool = essentia.Pool()

    framecutter.frame >> w.frame >> spectrum.frame
    spectrum.spectrum >> peaks.spectrum
    peaks.magnitudes >> hpcp.magnitudes
    peaks.frequencies >> hpcp.frequencies

    spectrum.spectrum >> gfcc.spectrum
    gfcc.bands >> None

    hpcp.hpcp >> (pool, 'lowlevel.hpcp')
    gfcc.gfcc >> (pool, fcc_name)
    loader = MonoLoader()

    loader.audio >> framecutter.signal

    for filename in ProgressBar(files):
        with h5py.File(track_file) as f:
            if filename not in f:
                try:
                    track_path = os.path.join(track_dir, filename)
                    loader.configure(filename=track_path,
                                     sampleRate=SAMPLE_RATE)

                    essentia.reset(loader)
                    essentia.run(loader)

                    grp = f.create_group(filename)
                    grp.create_dataset(TIMBRE_GROUP,
                                       data=pool[fcc_name],
                                       compression=9)
                    grp.create_dataset(CHROMA_GROUP,
                                       data=pool['lowlevel.hpcp'],
                                       compression=9)

                    bccs = beat_ffts_from_mfccs(pool[fcc_name])
                    grp.create_dataset(RHYTHM_GROUP, data=bccs, compression=9)

                    pool.clear()
                except (SystemExit, KeyboardInterrupt):
                    break
                except Exception as e:
                    logging.error(e)


def strip_incorrect_rhythm(track_file='../tracks.h5'):
    # progress = ProgressBar()
    with h5py.File(track_file) as f:
        keys = f.keys()
        for track in keys:
            grp = f[track]
            if RHYTHM_GROUP in grp and TIMBRE_GROUP in grp:
                del grp[RHYTHM_GROUP]


def add_rhythm(track_file='../tracks.h5'):
    progress = ProgressBar()
    with h5py.File(track_file, 'r') as f:
        keys = f.keys()
    for track in progress(keys):
        with h5py.File(track_file) as f:
            grp = f[track]
            if TIMBRE_GROUP in grp and RHYTHM_GROUP not in grp:
                bccs = beat_dcts_from_mfccs(grp[TIMBRE_GROUP])
                grp.create_dataset(RHYTHM_GROUP, data=bccs, compression=9)

if __name__ == '__main__':
    [track_dir, track_file] = sys.argv[1:]
    analyze_tracks(track_dir, track_file)
    # strip_incorrect_rhythm()
    # add_rhythm()
