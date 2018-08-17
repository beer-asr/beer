#!/bin/bash

if [ $# -ne 1 ];then
    echo "$0 <setup.sh>"
    exit 1
fi

setup=$1
. $setup
stage=0

if [ $stage -le 0 ];then
    echo "Decoding"
    mkdir -p $decode_dir
    python utils/decode_hmm.py \
        --gamma $hmm_gamma \
        $hmm_gmm_mdl_dir/final.mdl $decode_dir \
        $data_test_dir/feats.npz $hmm_emission_conf > $decode_dir/decode.log 2>&1 || exit 1
fi

if [ $stage -le 1 ];then
    echo "Merging states into phones"
fi

if [ $stage -le 2 ];then
    echo "Scoring"
fi
