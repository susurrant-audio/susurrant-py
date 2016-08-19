#!/usr/bin/env python

import os
import json
import h5py
import sys
from progressbar import ProgressBar
from itertools import groupby
from constants import valid_data_types
from run_vw import read_vw
from track_info import get_tracks


BASE_DIR = '../'
OUT_DIR = '../susurrant_elm/data'


def to_track(key):
    return key.split('#')[0]


def df_to_dict(df, combine=False):
    result = {}

    if combine:
        for key in df:
            track = to_track(key)
            if track not in result:
                result[track] = df[key]
            else:
                result[track] += df[key]
        return {k: (v / v.sum()).tolist() for k, v in result.iteritems()}
    else:
        for key in df:
            result[key] = df[key].tolist()
        return result


def df_to_json(df, filename, combine=False):
    d = df_to_dict(df, combine=combine)
    with open(filename, 'wb') as f:
        json.dump(d, f)


def topic_tokens_to_json(token_g_topic, filename):
    out = {}
    for i, key in enumerate(token_g_topic):
        row = token_g_topic[key].copy()
        row.sort(ascending=False)
        out[i] = row[0:10].to_dict()

    with open(filename, 'wb') as f:
        json.dump(out, f)


def all_valid(xs):
    return all(x is not None for x in xs)


def combine_tokens(grp):
    gfccs = grp.get('gfccs')
    chroma = grp.get('chroma')
    beat_coefs = grp.get('beat_coefs')
    if all_valid([gfccs, chroma, beat_coefs]):
        for i in xrange(len(gfccs)):
            beat_i = i // 256
            beat = beat_coefs[beat_i] if beat_i < len(beat_coefs) else None
            if beat is not None and i < len(chroma):
                token = {'beat_coef': beat,
                         'chroma': chroma[i],
                         'gfcc': gfccs[i]
                         }
                yield token


def check_if_valid_track(track_file, token_file_time):
    return (os.path.exists(track_file) and
            os.path.getmtime(track_file) >= token_file_time and
            os.path.getsize(track_file) > 3)


def save_track_tokens(token_file=os.path.join(BASE_DIR, 'vocab', 'tokens.h5'),
                      out_dir=OUT_DIR):
    progress = ProgressBar()
    token_file_time = os.path.getmtime(token_file)
    with h5py.File(token_file, 'r') as token_file:
        for track_filename in progress(token_file):
            track = track_filename.split('.')[0]
            track_file = os.path.join(out_dir, 'tracks', track + '.json')
            if not check_if_valid_track(track_file, token_file_time):
                with open(track_file, 'wb') as out:
                    tokens = []
                    grp = token_file[track_filename]
                    if set(grp.keys()) == valid_data_types:
                        tokens.extend(combine_tokens(grp))
                    json.dump(tokens, out)


def save_vw(vw_dir='../vw', out_dir=OUT_DIR):
    lda = read_vw(vw_dir)

    topics = os.path.join(out_dir, 'topics.json')
    doc_topics = os.path.join(out_dir, 'doc_topics.json')
    token_topics = os.path.join(out_dir, 'token_topics.json')
    topic_tokens = os.path.join(out_dir, 'topic_tokens.json')

    with open(topics, 'wb') as f:
        json.dump(lda.pr_topic.tolist(), f)

    topic_given_doc = lda.pr_topic_g_doc.T.dropna(how='all').T
    df_to_json(topic_given_doc, doc_topics, combine=True)

    topic_given_token = lda.pr_topic_g_token.T.dropna(how='all').T
    df_to_json(topic_given_token, token_topics)

    token_g_topic = lda.pr_token_g_topic.dropna(how='all')
    topic_tokens_to_json(token_g_topic, topic_tokens)


def save_bnpy(bnpy_dir='../bnpy', out_dir=OUT_DIR):
    model = read_bnpy(bnpy_dir)

    topics = os.path.join(out_dir, 'topics.json')
    doc_topics = os.path.join(out_dir, 'doc_topics.json')
    token_topics = os.path.join(out_dir, 'token_topics.json')
    topic_tokens = os.path.join(out_dir, 'topic_tokens.json')

    # with open(topics, 'wb') as f:
    #     json.dump(lda.pr_topic.tolist(), f)

    # topic_given_doc = lda.pr_topic_g_doc.T.dropna(how='all').T
    # df_to_json(topic_given_doc, doc_topics, combine=True)

    # topic_given_token = lda.pr_topic_g_token.T.dropna(how='all').T
    # df_to_json(topic_given_token, token_topics)

    # token_g_topic = lda.pr_token_g_topic.dropna(how='all')
    # topic_tokens_to_json(token_g_topic, topic_tokens)

def save_metadata(out_dir=OUT_DIR):
    metadata_file = os.path.join(out_dir, 'doc_metadata.json')
    doc_metadata = get_tracks()

    with open(metadata_file, 'wb') as f:
        json.dump(doc_metadata, f)


def main(seg_file, vw_dir, out_dir):
    save_vw(vw_dir, out_dir)
    save_track_tokens(seg_file, out_dir)
    # save_metadata()

if __name__ == '__main__':
    mode = sys.argv[1]
    out_dir = sys.argv[2]
    if mode == 'vw':
        vw_dir = sys.argv[3]
        save_vw(vw_dir, out_dir)
    elif mode == 'bnpy':
        bnpy_dir = sys.argv[3]
        save_bnpy(bnpy_dir, out_dir)
    elif mode == 'tracks':
        token_file = sys.argv[3]
        save_track_tokens(token_file, out_dir)
    elif mode == 'metadata':
        save_metadata(out_dir)
