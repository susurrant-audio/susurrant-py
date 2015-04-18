#!/usr/bin/env python

import h5py
import logging
import os
import sys
import numpy as np
from utils import TIMBRE_GROUP, valid_data_types
from progressbar import ProgressBar

logging.basicConfig(level=logging.INFO)


def training_for(data_type=TIMBRE_GROUP,
                 track_file='../tracks.h5',
                 training_file='../vocab/train',
                 ):
    progress = ProgressBar()
    if data_type not in valid_data_types:
        acceptable = '\n'.join(valid_data_types)
        raise Exception("Unrecognized data type! Must be one of:\n" +
                        acceptable)

    discard_power = data_type == TIMBRE_GROUP

    logging.info("Getting sizes of " + data_type)
    rows = 0
    cols = None
    with h5py.File(track_file, 'r') as f:
        for dset_name in progress(f):
            grp = f[dset_name]
            if data_type in grp:
                dset = grp[data_type]
                if cols is None:
                    y = dset.shape[1]
                    cols = (y - 1) if discard_power else y
                rows += dset.shape[0]

    print '{} x {}'.format(rows, cols)

    logging.info("Reading and outputting " + data_type)
    i = 0

    with h5py.File(training_file, 'w') as out:
        out.create_dataset("Y", data=np.zeros(rows), compression=9)
        X = out.create_dataset("X", (rows, cols), dtype='float32',
                               chunks=(8192, cols),
                               compression=9)
        with h5py.File(track_file, 'r') as f:
            for dset_name in progress(f):
                grp = f[dset_name]
                if data_type in grp:
                    if discard_power:
                        dset = grp[data_type][:, 1:]
                    else:
                        dset = grp[data_type]
                    if dset.shape[1] > 0:
                        dset_rows = dset.shape[0]
                        X[i:i+dset_rows, :] = dset
                        i += dset_rows

    return training_file


def main():
    for t in valid_data_types:
        training_for(t)

if __name__ == '__main__':
    [track_file, out_file] = sys.argv[1:]
    dtype = os.path.basename(out_file).replace('.h5', '')
    training_for(dtype, track_file, out_file)
