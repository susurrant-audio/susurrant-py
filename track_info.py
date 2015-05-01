#!/usr/bin/env python

import json
from py2neo import Graph

NODE_USER = "User"
NODE_TRACK = "Track"
NODE_COMMENT = "Comment"
NODE_PROFILE = "Profile"


NODE_PREFIXES = {x: '/' + x.lower() + '/' for x in
                 [NODE_USER, NODE_TRACK, NODE_COMMENT, NODE_PROFILE]}

REL_FOLLOWS = "FOLLOWS"
REL_UPLOADED = "UPLOADED"
REL_FAVORITED = "FAVORITED"
REL_HAS_PROFILE = "HAS_PROFILE"
REL_WROTE = "WROTE"
REL_REFERS_TO = "REFERS_TO"


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


def node_to_id(node):
    node_type = list(node.labels)[0]
    node_prefix = NODE_PREFIXES[node_type]
    node_id = node_prefix + str(node.properties['id'])
    return node_id


def get_graph():
    graph = Graph()
    nodes = set()
    links = []
    for rel_type in [REL_FOLLOWS, REL_UPLOADED, REL_FAVORITED]:
        for rel in graph.match(rel_type=rel_type):
            start = node_to_id(rel.start_node)
            end = node_to_id(rel.end_node)
            if rel_type == REL_FAVORITED and end not in nodes:
                continue
            nodes.add(start)
            nodes.add(end)
            links.append([start, end])
    return {'nodes': list(nodes), 'links': links}


if __name__ == '__main__':
    print get_tracks().values()[:10]
