#!/usr/bin/python

import sys
import os
import codecs
import subprocess
import shutil

MALLET_HOME = os.getenv("MALLET_HOME",
                        os.path.join(os.path.expanduser("~"),
                                     '/Applications/mallet-2.0.8RC2/'
                                     )
                        )
MALLET_BINARY = os.path.join(
    MALLET_HOME,
    '/bin/mallet')

def run_lda(INSTANCES_FILE=None, MALLET_OUT_DIR='../lda', TOPICS=100):
    if not os.path.exists(MALLET_OUT_DIR):
        os.makedirs(MALLET_OUT_DIR)
    included_instances = os.path.join(MALLET_OUT_DIR, 'instances.mallet')
    if INSTANCES_FILE is None:
        INSTANCES_FILE = included_instances
    if not os.path.exists(INSTANCES_FILE):
        raise Exception("No instance file provided!")

    PROGRESS_FILE = os.path.join(MALLET_OUT_DIR, 'progress.txt')

    mallet_opts = {
        'input': INSTANCES_FILE,
        'num-topics': TOPICS,
        'num-iterations': 1000,
        'optimize-interval': 10,
        'optimize-burn-in': 200,
        'use-symmetric-alpha': 'false',
        'alpha': 50.0,
        'beta': 0.01,
        'output-state': os.path.join(MALLET_OUT_DIR, 'topic-state.gz'),
        'output-doc-topics': os.path.join(MALLET_OUT_DIR, 'doc-topics.txt'),
        'output-topic-keys': os.path.join(MALLET_OUT_DIR, 'topic-keys.txt'),
        'word-topic-counts-file': os.path.join(MALLET_OUT_DIR,
                                               'word-topics.txt'),
        'diagnostics-file': os.path.join(MALLET_OUT_DIR,
                                         'diagnostics-file.txt'),
        'xml-topic-phrase-report': os.path.join(MALLET_OUT_DIR,
                                                'topic-phrases.xml'),
        }

    MALLET = [MALLET_BINARY, 'run']

    process_args = MALLET + ['cc.mallet.topics.tui.TopicTrainer']
    for (k, v) in mallet_opts.iteritems():
        process_args.append(u'--' + k)
        process_args.append(unicode(v))

    with codecs.open(PROGRESS_FILE, 'w', encoding='utf-8') as progress_file:
        subprocess.call(process_args,
                        stdout=progress_file,
                        stderr=progress_file)


if __name__ == '__main__':
    run_lda(*sys.argv[1:])
