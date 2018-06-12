#!/bin/bash


if [ $# -ne 1 ]; then
    echo "$0: setup.sh"
    exit 1
fi
setup=$1
. $setup


[ ! -d $hmm_model_dir ] && mkdir -p $hmm_model_dir && exit 1;

echo "Create emission models"
python3 steps/create_emission.py $nstates $feat_stats $hmm_model_dir \
    --emission_type $emission_type \
    --mean_normalize $mean_norm \
    --var_normalize $var_norm


echo "Train hmm models"
python3 -m cProfile -s cumtime steps/train_hmm.py $feats $labels $emissions \
    $feat_stats $hmm_model_dir \
    --mean_normalize $mean_norm \
    --var_normalize $var_norm \
    --context $context \
    --lrate $lrate \
    --batch_size $batch_size \
    --epochs $epochs
