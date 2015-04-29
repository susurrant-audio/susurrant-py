#!/usr/bin/env python

import json
from py2neo import Graph


def inflate(properties):
    """Turn Neo4j serialized dict into normal"""
    new_d = {}
    json_keys = [x for x in properties.keys()
                 if x.startswith("__json_")]
    for k in json_keys:
        orig_key = k[7:]
        new_d[orig_key] = json.loads(properties[orig_key])
    for k, v in properties.iteritems():
        if not k.startswith("__json_"):
            if k not in new_d:
                new_d[k] = v
    return new_d


def get_tracks():
    track_metadata = {}
    graph = Graph()

    for track in graph.find("Track"):
        track_metadata[track.properties['id']] = inflate(track.properties)
    return track_metadata

def get_track_comments():
    track_comments = {}
    graph = Graph()

    for comment in graph.find("Comment"):
        track_comments[comment.properties['id']] = inflate(comment.properties)
    return track_comments


if __name__ == '__main__':
    print get_tracks().values()[:10]
