#!/usr/bin/env python
from librosa.core import stft, istft
import numpy as np
import scipy

y = np.random.rand(44032)

stft_matrix = stft(y, window=scipy.signal.hann(2048), hop_length=1024)
y_hat = istft(stft_matrix, window=np.ones(2048), hop_length=1024)

diff = y - y_hat
print np.dot(diff, diff)
