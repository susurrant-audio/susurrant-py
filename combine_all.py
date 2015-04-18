#!/usr/bin/env python

import h5py
import numpy as np

mfccs = []

with h5py.File('tracks.hdf5', 'r') as f:
    grp = f.require_group("mfccs")
    for dset_name in grp:
        dset = grp[dset_name][1:, :]
        if dset.shape[1] > 0:
            mfccs.append(dset)

X = np.hstack(mfccs).T

with h5py.File('all_mfccs.hdf5') as f:
    f.create_dataset("mfccs", data=X)
