#!/usr/bin/python

import os
import glob
import subprocess
from cli import run_app
import sys
from rosetta.text import vw_helpers
from rosetta.text.text_processors import SFileFilter, VWFormatter


VW_EXECUTABLE = 'vw'
TOPICS = 10


def nlines(data_file):
    result = subprocess.check_output(['wc', '-l', data_file])
    return int(result.strip().split()[0])


def non_dupes_from(data_file):
    cmd = ('grep -v ":8192" ' + data_file +
           """ | awk '{gsub(/\|/,""); print $3;}'""")
    return subprocess.check_output(cmd, shell=True).split('\n')


def run_vw_lda(data_file='../vw/data.vw', topics=TOPICS):
    vw_dir = os.path.dirname(data_file)
    filtered_data_file = os.path.join(vw_dir, 'doc_tokens.vw')
    # num_docs = nlines(data_file)

    sff = SFileFilter(VWFormatter())
    sff.load_sfile(data_file)
#     sff.filter_extremes(doc_freq_min=3)
    sff.save(os.path.join(vw_dir, 'sff_file.pkl'))

    non_dupes = set(non_dupes_from(data_file))
    sff.filter_sfile(data_file, filtered_data_file,
#                     doc_id_list=non_dupes,
#                     enforce_all_doc_id=False,
                     min_tf_idf=0.3)

    progress_file = os.path.join(vw_dir, 'progress.txt')

    vw_opts = {
        '--lda': topics,
        # '--lda_alpha': 0.1,
        '--lda_rho': 0.5,
        # '--lda_D': num_docs,
        '--minibatch': 512,
        # '--power_t': 0.5,
        # '--initial_t': 1,
        '-b': 16,
        '--cache_file': os.path.join(vw_dir, 'ddrs.cache'),
        '--passes': 20,
        '-p': os.path.join(vw_dir, 'predictions.dat'),
        '--readable_model': os.path.join(vw_dir, 'topics.dat')
    }

    for fname in glob.glob(vw_dir + '/*cache'):
        os.unlink(fname)

    run_app(VW_EXECUTABLE, [filtered_data_file], vw_opts, progress_file)


def read_vw(vw_dir='../vw', topics=TOPICS):
    topics_file = os.path.join(vw_dir, 'topics.dat')
    prediction_file = os.path.join(vw_dir, 'predictions.dat')
    data_file = os.path.join(vw_dir, 'sff_file.pkl')
    return vw_helpers.LDAResults(topics_file,
                                 prediction_file,
                                 data_file,
                                 num_topics=topics)

if __name__ == '__main__':
    run_vw_lda(*sys.argv[1:])
