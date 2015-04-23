#!/usr/bin/env python

import h5py
import sys
import os
from progressbar import ProgressBar
from constants import valid_data_types


def pieces(xs, per_piece):
    ndims = len(xs.shape)
    total = xs.shape[0]
    i = 0
    while i + per_piece < total:
        if ndims == 2:
            yield xs[i:i+per_piece, :]
        else:
            yield xs[i:i+per_piece]
        i += per_piece
    if ndims == 2:
        yield xs[i:, :]
    else:
        yield xs[i:]


def segment_one(track_name, grp, out, frames_per_segment=8192):
    beat_frames = frames_per_segment / 256
    for dtype in valid_data_types:
        if dtype == 'beat_coefs':
            segs = pieces(grp[dtype], beat_frames)
        else:
            segs = pieces(grp[dtype], frames_per_segment)
        for i, seg in enumerate(segs):
            grp_name = '{}#{:#06x}'.format(track_name, i)
            out_grp = out.require_group(grp_name)
            out_grp.create_dataset(dtype, data=seg, compression=9)


def segment(track_file='../tracks.h5', out_file='../segmented.h5'):
    progress = ProgressBar()
    partway = (os.path.exists(out_file) and
               os.path.getmtime(track_file) < os.path.getmtime(out_file))

    with h5py.File(out_file) as out:
        done = set([x.split('#')[0] for x in out.keys()])

        with h5py.File(track_file, 'r') as f:
            for track in progress(f):
                if track not in done and not partway:
                    grp = f[track]
                    if set(grp.keys()) == valid_data_types:
                        segment_one(track, grp, out)

if __name__ == '__main__':
    segment(*sys.argv[1:])
