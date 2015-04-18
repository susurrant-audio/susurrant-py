#!/usr/bin/env /Users/chrisjr/Development/susurrant_prep/py/my_jython.sh

from cc.mallet.types import *
from cc.mallet.pipe import *

from java.io import ObjectInputStream, FileInputStream

from constants import valid_data_types


def readInstanceList(fname):
    s = ObjectInputStream(FileInputStream(fname))
    instances = s.readObject()
    s.close()
    return instances


def instances_to_json(MALLET_FILE='../track_tokens.mallet',
                      out_dir='../susurrant_elm/data/tracks/'):
    instances = readInstanceList(MALLET_FILE)

    for instance in instances.iterator():
        track = instances.getName()
        print track
        break
        track_data = []
        dtypes = h5.object().getGroupMembers('/' + track)
        if set(dtypes) != valid_data_types:
            continue
        for dtype in valid_data_types:
            dset = "/{}/{}".format(track, dtype)
            data = h5.readIntArray(dset)
            track_data.extend([dtype + str(x) for x in data])
        ts = TokenSequence(track_data)
        instance_list.add(Instance(ts, None, track, None))

    instances.addThruPipe(instance_list.iterator())

    saveInstanceList(instances, MALLET_FILE)

if __name__ == '__main__':
    tokens_to_instances()
