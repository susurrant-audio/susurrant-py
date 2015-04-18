import h5py

track_file = '../tracks.h5'

i = 0.0
total = 0

with h5py.File(track_file, 'r') as f:
    for key in f:
        total += 1
        if 'beat_coefs' in f[key]:
            i += 1

print i / total
