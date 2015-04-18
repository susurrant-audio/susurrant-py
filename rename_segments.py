#!/usr/bin/env python

import h5py
from progressbar import ProgressBar


def rename_segments(track_file='../segmented.h5'):
    progress = ProgressBar()
    with h5py.File(track_file) as f:
        keys = f.keys()
        for key in progress(keys):
            parts = key.split('#')
            track_name = parts[0]
            i = int(parts[1])
            seg_name = '{}#{:#06x}'.format(track_name, i)
            f.move(key, seg_name)

if __name__ == '__main__':
    rename_segments()
