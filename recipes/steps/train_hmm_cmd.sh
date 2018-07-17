#!/bin/bash


if [ $# -ne 1 ]; then
    echo "$0: setup.sh"
    exit 1
fi
setup=$1
. $setup

python3 steps/train_hmm.py $feats $labels $emissions \
    $feat_stats $hmm_model_dir $use_gpu \
    --training_type $training_type \
    --lrate $lrate \
    --batch_size $batch_size \
    --epochs $epochs
