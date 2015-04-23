import os
import h5py
import numpy as np
from constants import valid_data_types, TIMBRE_GROUP
from track_to_kmean_training import training_for
from tempfile import NamedTemporaryFile


def temp_path():
    f = NamedTemporaryFile(delete=False)
    path = f.name
    os.unlink(path)
    return path


def generate_tracks(n, m=100):
    tracks = []
    for i in xrange(n):
        dsets = {k: np.random.rand(m, 10) for k in valid_data_types}
        tracks.append((str(i), dsets))
    return tracks


def make_tracks(tracks):
    fname = temp_path()
    with h5py.File(fname, 'w') as f:
        for track_name, dsets in tracks:
            grp = f.require_group(track_name)
            for name, dset in dsets.iteritems():
                grp.create_dataset(name, data=dset)
    return fname


def combine_tracks(tracks, dtype):
    names, dsets = zip(*tracks)
    return np.vstack(map(lambda x: x.get(dtype), dsets))


def test_training_for(tracks_n=10):
    track_values = generate_tracks(tracks_n)
    track_file = make_tracks(track_values)
    result_files = {}
    for dtype in valid_data_types:
        out_file = temp_path()
        training_for(dtype, track_file, out_file)
        result_files[dtype] = out_file

    for dtype, fname in result_files.iteritems():
        combined = combine_tracks(track_values, dtype)
        if dtype == TIMBRE_GROUP:
            combined = combined[:, 1:]
        with h5py.File(fname, 'r') as f:
            data = f['/X'].value
            if tracks_n > 10:
                print combined, data
            assert data.shape == combined.shape
            print np.absolute(data - combined).sum()
            assert np.absolute(data - combined).sum() < 1e-3

    os.unlink(track_file)
    for fname in result_files.values():
        os.unlink(fname)


def test_training_for_large():
    test_training_for(82)
