import h5py
from progressbar import ProgressBar

with h5py.File('../tracks_bak.h5', 'r') as f:
    progress = ProgressBar()
    with h5py.File('../tracks.h5') as out:
        for track in progress(f):
            grp = f[track]
            out_grp = out.create_group(track)
            for name, data in grp.iteritems():
                out_grp.create_dataset(name, data=data, compression=9)
