#!/usr/bin/env python

import h5py
import logging
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
# from sklearn.datasets import dump_svmlight_file
from progressbar import ProgressBar

logging.basicConfig(level=logging.INFO)

progress = ProgressBar()

mfccs = []

logging.info("Reading MFCCs")

with h5py.File('tracks.hdf5', 'r') as f:
    grp = f.require_group("mfccs")
    for dset_name in progress(grp):
        dset = grp[dset_name][1:, :]
        if dset.shape[1] > 0:
            mfccs.append(dset)

X = np.hstack(mfccs).T

logging.info("Calculating clusters")

with h5py.File('mfccs.h5') as f:
    f.create_dataset("X", data=X)
    f.create_dataset("Y", data=np.zeros(X.shape[0]))
# dump_svmlight_file(X, np.zeros(X.shape[0]), "mfccs.train", zero_based=False)

if False:
    kmeans = MiniBatchKMeans(n_clusters=10000)

    start = time.clock()
    kmeans.fit(X)

    print("{} seconds".format(time.clock() - start))
    print(kmeans.score(X))
