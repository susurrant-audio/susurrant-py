import logging
from utils import TIMBRE_GROUP, RHYTHM_GROUP, CHROMA_GROUP
from track_to_kmean_training import training_for
import os


logging.basicConfig(level=logging.INFO)

data_types = {
    TIMBRE_GROUP: 5000,
    RHYTHM_GROUP: 2500,
    CHROMA_GROUP: 24
}


def do_train(track_file='../tracks.h5'):
    from sofia_kmeans import cluster

    all_opts = []
    for t, k in data_types.iteritems():
        training_file = training_for(t, track_file=track_file)
        opts = {'training_file': training_file, 'k': k,
                'out_dir': os.path.dirname(training_file)}
        all_opts.append(opts)
    cluster.map(all_opts)

def train_for()

if __name__ == '__main__':
    do_train()
