#!/usr/bin/env python

import h5py

mfccs = []

with h5py.File('tracks.hdf5', 'r') as f:
    with h5py.File('tracks_no_power.h5', 'w') as g:
        grp = f.require_group("mfccs")
        grp_out = g.require_group("mfccs")
        for dset_name in grp:
            dset = grp[dset_name][1:, :]
            if dset.shape[0] == 12 and dset.shape[1] > 1:
                grp_out.create_dataset(dset_name, data=dset.T)
