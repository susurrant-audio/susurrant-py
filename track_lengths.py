#!/usr/bin/env python

import eyed3
import os
import wave
import numpy as np
import h5py
import os
from contextlib import closing

track_file = '../tracks.h5'
track_dir = os.path.join(os.path.expanduser('~'),
                         '/Desktop/tracks')

times = []
frames = []

with h5py.File(track_file, 'r') as f:
    for track in f:
        track_path = os.path.join(track_dir, track)
        if track_path.endswith('.mp3'):
            t = eyed3.load(track_path).info.time_secs
        elif track_path.endswith('.wav'):
            with closing(wave.open(track_path, 'r')) as w:
                t = w.getnframes() * 1.0 / w.getframerate()
        times.append(t)
        grp = f.require_group(track)
        if 'gfccs' in grp:
            frames.append(grp['gfccs'].shape[0])
        else:
            frames.append(0)

timeframes = np.asarray([times, frames])
