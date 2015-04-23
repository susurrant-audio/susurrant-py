import numpy as np
import os
import h5py
from h5utils import by_chunk
from constants import valid_data_types, TIMBRE_GROUP
from tracks_to_assignments import get_anns
from collections import Counter


def non_zeros(arr):
    return np.count_nonzero(np.any(np.asarray(arr), axis=1))


def check_kmeans(cluster_file):
    if os.path.exists(cluster_file):
        vectors = np.loadtxt(cluster_file)
        nnz = non_zeros(vectors)
        return float(nnz) / vectors.shape[0]
    else:
        return 0.0


def test_kmeans():
    for dtype in valid_data_types:
        res = check_kmeans('../vocab/train/clusters_{}.txt'.format(dtype))
        assert res > 0.95


def test_tokens():
    anns = get_anns()

    def get_nns(dtype, values):
        if dtype == TIMBRE_GROUP:
            values = values[:, 1:]
        tree = anns[dtype]
        assigned = np.asarray([tree.nn(x) for x in values])
        return assigned

    def check_tokens(track_file, token_file, track=None):
        raw_by_dtype = {}
        results = {}
        with h5py.File(track_file, 'r') as tracks:
            if track is None:
                track = tracks.keys()[0]
            grp = tracks[track]
            for dtype in valid_data_types:
                raw_by_dtype[dtype] = grp[dtype].value
        with h5py.File(token_file, 'r') as tokens:
            grp = tokens[track]
            for dtype, values in raw_by_dtype.iteritems():
                dtype_tokens = grp[dtype].value
                nns = get_nns(dtype, values)
                equals = dtype_tokens == nns
                results[dtype] = float(equals.sum()) / len(equals)
        return results

    results = check_tokens('../tracks.h5', '../vocab/tokens.h5')
    assert all([x > 0.95 for x in results.values()])


def non_zero_rows(h5_file):
    with h5py.File(h5_file, 'r') as f:
        nnz = 0
        for chunk in by_chunk(f['/X']):
            nnz += non_zeros(chunk)
        return float(nnz)/f['/X'].shape[0]


def test_dtype_h5s():
    for dtype in valid_data_types:
        fname = '../vocab/train/{}_sampled.h5'.format(dtype)
        with h5py.File(fname, 'r') as f:
            res = float(non_zeros(f['/X'])) / f['/X'].shape[0]
            assert res > 0.95


def count_types(grp):
    results = {}
    for dtype in valid_data_types:
        x = grp[dtype]
        y = np.bincount(x)
        ii = np.nonzero(y)[0]
        results[dtype] = Counter(dict(zip(ii, y[ii])))
    return results


def merge_counters(x, y):
    assert set(x.keys()) == set(y.keys())
    result = {}
    for k in x.keys():
        result[k] = x[k] + y[k]
    return result


def test_segmented_h5():
    originals = {}
    with h5py.File('../vocab/tokens.h5', 'r') as f:
        valid_tracks = len(f.keys())
        sample_tracks = np.random.choice(f.keys(), 15)
        for k in sample_tracks:
            grp = f[k]
            if set(grp.keys()) == valid_data_types:
                originals[k] = count_types(grp)

    with h5py.File('../segmented.h5', 'r') as f:
        assert len(f.keys()) >= valid_tracks
        for k, orig in originals.iteritems():
            segments = [f[x] for x in f.keys() if x.startswith(k)]
            assert all(set(x.keys()) == valid_data_types
                       for x in segments)
            counts = [count_types(grp) for grp in segments]
            combined = reduce(merge_counters, counts)
            assert combined == orig
            assert counts[0]['gfccs'].values()[0] != 8192
