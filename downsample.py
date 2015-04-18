#!/usr/bin/env python

import h5py
import numpy as np
import sys
from progressbar import ProgressBar


def downsample(data_file='../vocab/train/chroma.h5', out_file=None,
               max_samples=2**24):
    if out_file is None:
        out_file = data_file.replace('.h5', '_sampled.h5')
    with h5py.File(data_file, 'r') as f:
        dset = f['/X']
        rows_per_chunk = dset.chunks[0]
        chunks = dset.shape[0] // rows_per_chunk
        samples_per_chunk = max_samples // chunks
        total_rows = samples_per_chunk * chunks
        print samples_per_chunk, total_rows
        progress = ProgressBar()
        with h5py.File(out_file, 'w') as out:
            out.create_dataset("Y", data=np.zeros(total_rows))
            X = out.create_dataset("X",
                                   (total_rows, dset.chunks[1]),
                                   chunks=(samples_per_chunk, dset.chunks[1]))
            for i in progress(range(chunks)):
                k = i * rows_per_chunk
                idxs = np.random.random_integers(0, rows_per_chunk - 1,
                                                 samples_per_chunk)
                idxs.sort()
                block = dset[k:k+rows_per_chunk, :]
                res = block[idxs, :]
                samples_per_chunk = min(res.shape[0], samples_per_chunk)
                X[k:k+samples_per_chunk, :] = res

if __name__ == '__main__':
    downsample(*sys.argv[1:])
