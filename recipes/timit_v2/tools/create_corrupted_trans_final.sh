#!/bin/bash

ser=0.3
error_prob=0.2
error_type="all"
sub_list="data/lang/create_corrupted_testset/phone_confusion.list"
ins_list="data/lang/create_corrupted_testset/ins.list"
input_dir="data/dev"
output_dir="data/dev_${error_type}_${error_prob}"

if [ ! -f $output_dir ];then
    mkdir -p $output_dir
fi

cp $input_dir/{feats.npz,trans,uttids} $output_dir

python tools/create_corrupted_trans_final.py \
        --error_type $error_type \
        --error_prob $error_prob \
        --ser $ser \
        --sub_map $sub_list \
        --ins_list $ins_list \
        $input_dir/trans \
        $output_dir/trans


