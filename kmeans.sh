#!/bin/bash

kmeans_train() {
    data_type="$1"
    k="$2"
    if [ $data_type = "beat_coefs" ]; then
        d="13"
        k="${k:-2500}"
    elif [ $data_type = "chroma" ]; then
        d="13"
        k="${k:-24}"
    elif [ $data_type = "gfccs" ]; then
        d="13"
        k="${k:-5000}"
    fi
    training_file="$HOME/train/${data_type}.h5"
    out_file="$HOME/train/clusters_${data_type}.txt"

    if [ ! -e $out_file ]; then
        $HOME/src/sofia-ml/sofia-kmeans \
                --k $k \
                --init_type random \
                --opt_type mini_batch_kmeans \
                --mini_batch_size 100 \
                --iterations 500 \
                --objective_after_training \
                --training_file_h5 $training_file \
                --dimensionality $d \
                --model_out $out_file
    fi
}

kmeans_train $1