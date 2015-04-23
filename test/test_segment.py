import numpy as np
from segment import *


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
