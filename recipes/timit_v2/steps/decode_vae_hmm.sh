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
    mkdir -p $vae_hmm_decode_dir
    python utils/decode_vae_hmm.py \
        --gamma $hmm_gamma \
        $vae_hmm_dir/final.mdl $vae_hmm_decode_dir \
        $data_test_dir/feats.npz $vae_hmm_emissions_conf \
        > $vae_hmm_decode_dir/decode.log 2>&1 || exit 1
fi

if [ $stage -le 1 ];then
    echo "Merging states into phones"
fi

if [ $stage -le 2 ];then
    echo "Scoring"
fi
