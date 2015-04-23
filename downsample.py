#!/usr/bin/env python

import h5py
import numpy as np
import sys
from h5utils import by_chunk


def downsample(data_file='../vocab/train/chroma.h5', out_file=None,
               max_samples=2**16):  # 2**24):
    if out_file is None:
        out_file = data_file.replace('.h5', '_sampled.h5')
    with h5py.File(data_file, 'r') as f:
        dset = f['/X']
        rows_per_chunk = dset.chunks[0]
        chunks = dset.shape[0] // rows_per_chunk
        samples_per_chunk = max_samples // chunks
        total_rows = samples_per_chunk * chunks
        print '{}/chunk for {} chunks: {} samples'.format(samples_per_chunk,
                                                          chunks,
                                                          total_rows)
        with h5py.File(out_file, 'w') as out:
            out.create_dataset("Y", data=np.zeros(total_rows))
            X = out.create_dataset("X",
                                   (total_rows, dset.chunks[1]),
                                   chunks=(samples_per_chunk, dset.chunks[1]))
            k = 0
            for block in by_chunk(dset):
                idxs = np.random.random_integers(0, rows_per_chunk - 1,
                                                 samples_per_chunk)
                idxs.sort()
                res = block[idxs, :]
                samples_per_chunk = min(res.shape[0], samples_per_chunk)
                X[k:k+samples_per_chunk, :] = res
                k += samples_per_chunk

if __name__ == '__main__':
    downsample(*sys.argv[1:])
