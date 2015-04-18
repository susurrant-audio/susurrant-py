import os
import subprocess
import logging

from constants import valid_data_types

from proc_all_tracks import analyze_tracks
from train_kmeans import do_train
from index_clusters import create_index
from tracks_to_assignments import tracks_to_assignments
from run_lda import run_lda

BASE_DIR = '/Users/chrisjr/Development/susurrant_prep'


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("")

print "Please run: ipcluster start -n 2"


def stale(filename, track_file='../tracks.h5'):
    f_mtime = os.path.getmtime(filename)
    track_mtime = os.path.getmtime(track_file)
    return f_mtime < track_mtime


def main():
    track_file = '../tracks.h5'

    if not os.path.exists(track_file):
        track_dir = '/Users/chrisjr/Desktop/tracks'
        logger.info("Analyzing tracks in {}".format(track_dir))
        analyze_tracks(track_file, track_dir)

    cluster_file = '../vocab/train/clusters_chroma.txt'
    if not os.path.exists(cluster_file):
        logger.info("Training k-means")
        do_train(track_file)

    tree_file = '../vocab/train/clusters_chroma.tree'
    if not os.path.exists(tree_file):
        logger.info("Training ANNs")
        for data_type in valid_data_types:
            fname = 'clusters_{}.txt'.format(data_type)
            cluster_file = os.path.join(BASE_DIR, 'vocab', 'train', fname)
            create_index(cluster_file)

    assignment_file = '../vocab/tokens.h5'
    if not os.path.exists(assignment_file):
        logger.info("Turning tracks into tokens")
        tracks_to_assignments()

    instance_file = '../track_tokens.mallet'
    if not os.path.exists(instance_file):
        logger.info("Generating instance file")
        subprocess.call('tokens_to_instances.py')  # this is jython

    state_file = '../lda/topic-state.gz'
    if not os.path.exists(state_file):
        logger.info("Training lda")
        run_lda()

    logger.info("Complete.")

if __name__ == '__main__':
    main()
