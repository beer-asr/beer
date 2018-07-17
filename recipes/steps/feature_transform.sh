#!/bin/bash

if [ $# != 2 ]; then
    echo "$0: data_dir feature.conf"
    exit 1
fi

data_dir=$1
feat_conf=$2

. $feat_conf
cp $feat_conf $data_dir

if [ ! -f $data_dir/feats.npz ];then
    echo "Original feature does not exist"
    exit 1
else
    mkdir -p $data_dir/feat_transform
    python3 steps/feature_transform.py $data_dir $data_dir/feat_transform/ \
            $mean_norm $var_norm $add_delta \
            --context $context \
            --norm_type $norm_type
    zip -j $data_dir/feats_transformed.npz $data_dir/feat_transform/*.npy
    rm -r $data_dir/feat_transform
fi


