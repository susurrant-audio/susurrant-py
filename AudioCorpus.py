''' AudioCorpus.py
    The GFCC data from SoundCloud.
'''
import bnpy.data.GroupXData as GroupXData
import h5py
from ipy_progressbar import ProgressBar
import numpy as np
import os

DATAFILE_MAT = 'audiocorpus.mat'

def get_data(**kwargs):
    ''' Returns data from audio tracks
    '''
    
    if os.path.exists(DATAFILE_MAT):
        Data = GroupXData.LoadFromFile(DATAFILE_MAT)
    else:
        obs = []
        doc_range = [0]
        count = 0
        with h5py.File('../tracks.h5', 'r') as tracks:
            for track, grp in ProgressBar(tracks.items()):
                if 'gfccs' not in grp:
                    continue
                data = grp['gfccs']
                count += data.shape[0]
                doc_range.append(count)
                obs.append(data.value.astype(np.float64))
        X = np.vstack(obs)
        Data = GroupXData(X=X, doc_range=doc_range)
        Data.save_to_mat(DATAFILE_MAT)
    Data.name = 'AudioCorpus'
    Data.summary = 'Audio Corpus. obs=10.5M docs=559'

    return Data

if __name__ == '__main__':
    get_data()