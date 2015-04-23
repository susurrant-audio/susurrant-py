#!/usr/bin/env python
import sys
import glob
import os
import re

CLUSTER_SIZE_RE = re.compile(r"# Cluster size: (\d+)")
VECTOR_RE = re.compile(r"ID=(\d+) ([0-9.-]+\s?)+")


def parse_dbscan(dbscan_dir):
    result = {'clusters': []}
    for cluster_file in glob.glob(os.path.join(dbscan_dir, 'cluster_*.txt')):
        cluster = {'exemplar': None, 'count': 0, 'ids': []}
        with open(cluster_file, 'r') as f:
            size_found = False
            exemplar_found = False
            for line in f:
                if not size_found:
                    size_match = CLUSTER_SIZE_RE.match(line)
                    if size_match is not None:
                        size_found = True
                        cluster['count'] = int(size_match.group(1))
                else:
                    vec_match = VECTOR_RE.match(line)
                    if vec_match is not None:
                        vec_id = vec_match.group(1)
                        cluster['ids'].append(int(vec_id))
                        if not exemplar_found:
                            vec = vec_match.groups()[1:]
                            exemplar_found = True
                            cluster['exemplar'] = [float(x) for x in vec]
        result['clusters'].append(cluster)
    return result

if __name__ == '__main__':
    parse_dbscan(sys.argv[1])
