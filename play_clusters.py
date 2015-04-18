#!/usr/bin/env python

from utils import imfcc
import numpy as np
from scikits.audiolab import wavwrite, play

CLUSTER_FILE = '../old_data/kmeans/clusters_rand_5000.txt'
PLAYING = False

clusters = np.loadtxt(CLUSTER_FILE)[:, 1:]


for cluster in clusters:
    x = np.tile(cluster, (51, 1))
    y = imfcc(x, has_power=False)
    if PLAYING:
        play(y, fs=44100)
    else:
        wavwrite(y, 'test.wav', fs=44100, enc='pcm16')
        break
