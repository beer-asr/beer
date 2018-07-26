#!/bin/bash


if [ $# -ne 1 ]; then
    echo "$0: setup.sh"
    exit 1
fi
setup=$1
. $setup

echo "Creating models in $hmm_model_dir"
export CUDA_VISIBLE_DEVICES=$(free-gpu)
echo $(hostname)
echo $CUDA_VISIBLE_DEVICES
python3 -m cProfile -s cumtime steps/train_hmm.py $feats $labels $emissions \
    $feat_stats $hmm_model_dir $use_gpu $fast_eval \
    --training_type $training_type \
    --lrate $lrate \
    --batch_size $batch_size \
    --epochs $epochs || exit 1
