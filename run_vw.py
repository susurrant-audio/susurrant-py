#!/usr/bin/python

import os
import glob
import codecs
import subprocess
import sys
from rosetta.text import vw_helpers
from rosetta.text.text_processors import SFileFilter, VWFormatter


VW_EXECUTABLE = 'vw'
TOPICS = 10


def nlines(data_file):
    result = subprocess.check_output(['wc', '-l', data_file])
    return int(result.strip().split()[0])


def run_vw_lda(data_file='../vw/data.vw', topics=TOPICS):
    vw_dir = os.path.dirname(data_file)
    filtered_data_file = os.path.join(vw_dir, 'doc_tokens.vw')
    # num_docs = nlines(data_file)

    sff = SFileFilter(VWFormatter())
    sff.load_sfile(data_file)
    sff.save(os.path.join(vw_dir, 'sff_file.pkl'))
    sff.filter_sfile(data_file, filtered_data_file)

    progress_file = os.path.join(vw_dir, 'progress.txt')

    vw_opts = {
        '--lda': topics,
        # '--lda_alpha': 0.1,
        # '--lda_rho': 0.1,
        # '--lda_D': num_docs,
        # '--minibatch': 256,
        # '--power_t': 0.5,
        # '--initial_t': 1,
        '-b': 16,
        '--cache_file': os.path.join(vw_dir, 'ddrs.cache'),
        '--passes': 10,
        '-p': os.path.join(vw_dir, 'predictions.dat'),
        '--readable_model': os.path.join(vw_dir, 'topics.dat')
    }

    process_args = [VW_EXECUTABLE, filtered_data_file]

    for (k, v) in vw_opts.iteritems():
        process_args.append(k)
        process_args.append(unicode(v))

    for fname in glob.glob(vw_dir + '/*cache'):
        os.unlink(fname)

    with codecs.open(progress_file, 'w', encoding='utf-8') as prog:
        subprocess.call(process_args,
                        stdout=prog,
                        stderr=prog)


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
