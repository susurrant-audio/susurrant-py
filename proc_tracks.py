#!/usr/bin/env python

import os
import h5py
import logging
from essentia.standard import *
from progressbar import ProgressBar
from utils import (WINDOW_SIZE, HOP_SIZE, SAMPLE_RATE, USE_GFCC,
                   TIMBRE_GROUP, RHYTHM_GROUP,
                   LOW_FREQ, HIGH_FREQ, BANDS, COEFS)

progress = ProgressBar()

TRACK_DIR = '/Users/chrisjr/Desktop/tracks'
files = [x for x in os.listdir(TRACK_DIR) if x != '.DS_Store']

w = Windowing(type='hann', size=WINDOW_SIZE)
spectrum = Spectrum()
options = {'sampleRate': SAMPLE_RATE,
           'numberBands': BANDS,
           'numberCoefficients': COEFS,
           'lowFrequencyBound': LOW_FREQ,
           'highFrequencyBound': HIGH_FREQ
           }
mfcc = MFCC(**options)
gfcc = GFCC(**options)

with h5py.File('../tracks.h5', 'w') as f:
    timbre_grp = f.require_group(TIMBRE_GROUP)
    rhythm_grp = f.require_group(RHYTHM_GROUP)
    for filename in progress(files[8:12]):
        try:
            loader = MonoLoader(filename=os.path.join(TRACK_DIR, filename),
                                sampleRate=SAMPLE_RATE)
            audio = loader()
            mfccs = []
            gen = FrameGenerator(audio, frameSize=WINDOW_SIZE,
                                 hopSize=HOP_SIZE)
            for i, frame in enumerate(gen):
                if USE_GFCC:
                    mfcc_bands, mfcc_coeffs = gfcc(spectrum(w(frame)))
                else:
                    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
                mfccs.append(mfcc_coeffs)
            mfccs = essentia.array(mfccs)
            timbre_grp.create_dataset(filename, data=mfccs)
            # rhythm_grp.create_dataset(filename, data=rhythm_matrix)
        except SystemExit, KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(e)


# add
# e.g.
#     tracks.update({"filename": filename}, {"$set": {"mfcc": x}}, upsert=True)
