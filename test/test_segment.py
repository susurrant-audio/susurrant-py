import numpy as np
import h5py
import os
from segment import *
from test.test_track_to_kmean_training import (generate_tracks,
                                               make_tracks,
                                               temp_path)


def test_pieces():
    sample_1d = np.random.rand(101)
    pieces_1d = list(pieces(sample_1d, 10))
    assert pieces_1d[0].shape == (10,)
    assert pieces_1d[-1].shape == (1,)
    assert len(pieces_1d) == 11
    recombined_1d = np.concatenate(pieces_1d)
    assert np.array_equal(sample_1d, recombined_1d)

    sample_2d = np.random.rand(101, 10)
    pieces_2d = list(pieces(sample_2d, 10))
    assert len(pieces_2d) == 11
    assert pieces_2d[0].shape == (10, 10)
    assert pieces_2d[-1].shape == (1, 10)
    recombined_2d = np.vstack(pieces_2d)
    assert np.array_equal(recombined_2d, sample_2d)


def test_segment():
    track_values = generate_tracks(10, 301)
    track_file = make_tracks(track_values)
    track_values = dict(track_values)

    segmented_file = temp_path()
    new_track_file = temp_path()
    segment(track_file, segmented_file, 100)

    with h5py.File(segmented_file, 'r') as f:
        for track, segs in groupby(f, lambda x: x.split('#')[0]):
            assert len(list(segs)) == 4

    rejoin(segmented_file, new_track_file)

    with h5py.File(new_track_file, 'r') as f:
        for track, dsets in f.iteritems():
            for dtype, dset in dsets.iteritems():
                assert np.array_equal(
                    track_values[track][dtype],
                    dset)

    for fname in [track_file, segmented_file, new_track_file]:
        os.unlink(fname)
