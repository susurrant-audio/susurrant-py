#!/usr/bin/env /Users/chrisjr/Development/susurrant_prep/py/my_jython.sh

from ch.systemsx.cisd.hdf5 import HDF5Factory

from cc.mallet.types import *
from cc.mallet.pipe import *

from java.io import ObjectOutputStream, FileOutputStream
from java.util import ArrayList

from constants import valid_data_types


def saveInstanceList(instances, fname):
    s = ObjectOutputStream(FileOutputStream(fname))
    s.writeObject(instances)
    s.close()


def tokens_to_instances(MALLET_FILE='../track_tokens.mallet',
                        token_file='../vocab/tokens.h5'):
    h5 = HDF5Factory.openForReading(token_file)

    ts2fs = TokenSequence2FeatureSequence()
    instances = InstanceList(ts2fs)

    instance_list = ArrayList()

    for track in h5.object().getGroupMembers('/'):
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
