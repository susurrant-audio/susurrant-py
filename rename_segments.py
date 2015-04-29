#!/usr/bin/env python

import h5py
from progressbar import ProgressBar


def rename_segments(in_file='../segmented.h5.bak', out_file='../segmented.h5'):
    progress = ProgressBar()
    with h5py.File(in_file, 'r') as f:
        with h5py.File(out_file, 'w') as out:
            keys = f.keys()
            for key in progress(keys):
                parts = key.split('#')
                track_name = parts[0]
                track_id = track_name.split('.')[0]
                i = int(parts[1], 16)
                seg_name = '{}#{:#06x}'.format(track_id, i)
                out.copy(f[key], out, name=seg_name)

if __name__ == '__main__':
    rename_segments()
