#!/usr/bin/env python

from utils import (imfcc, igfcc, SAMPLE_RATE, USE_GFCC,
                   TIMBRE_GROUP)
import h5py
import numpy as np
from scikits.audiolab import wavwrite, play

PLAYING = False
HAS_POWER = True
OUT_FILE = 'test_gfcc.wav' if USE_GFCC else 'test_mfcc.wav'


with h5py.File('../tracks.h5', 'r') as f:
    grp = f.require_group(TIMBRE_GROUP)
    k = grp.keys()[1]
    print k
    dset = grp[k].value

if USE_GFCC:
    y = igfcc(dset, has_power=HAS_POWER)
else:
    y = imfcc(dset, has_power=HAS_POWER)


if SAMPLE_RATE == 22050:
    y = np.repeat(y, 2)

if PLAYING:
    play(y, fs=44100)
else:
    wavwrite(y, OUT_FILE, fs=44100, enc='pcm16')
