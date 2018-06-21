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
    python3 steps/feature_transform.py $data_dir \
            $mean_norm $var_norm $add_delta \
            --context $context \
            --norm_type $norm_type
fi


