#!/usr/bin/env python
import h5py
import numpy as np
import sys
from progressbar import ProgressBar


def by_chunk(dset):
    progress = ProgressBar()
    chunk_size = dset.chunks[0]
    chunks = dset.shape[0] // chunk_size
    for i in progress(range(chunks)):
        k = i * chunk_size
        yield dset[k:k+chunk_size, :]


def non_zeros(arr):
    return np.count_nonzero(np.any(np.asarray(arr), axis=1))


def non_zero_rows(h5_file):
    with h5py.File(h5_file, 'r') as f:
        nnz = 0
        for chunk in by_chunk(f['/X']):
            nnz += non_zeros(chunk)
        return float(nnz)/f['/X'].shape[0]

if __name__ == '__main__':
    non_zero_rows(sys.argv[1])
