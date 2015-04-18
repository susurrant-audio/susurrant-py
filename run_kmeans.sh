#!/bin/bash

for i in "beat_coefs" "chroma" "gfccs"; do
    qsub ~/kmeans.sh $i
done