#!/usr/bin/env python
import json
import sys
from track_info import get_graph


def gen_graph(graph_file='../susurrant_elm/data/graph.json'):
    graph = get_graph()

    with open(graph_file, 'wb') as f:
        json.dump(graph, f)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        gen_graph(sys.argv[1])
    else:
        gen_graph()
