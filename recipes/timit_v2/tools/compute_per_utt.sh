#!/bin/bash

uttids="data/test_50_wrong_trans/uttids"
trans="data/test_50_wrong_trans/trans"
hyp="comp_llhs_new/test_50_wrong_trans/hyp_50_wrong"
resultdir="icassp_task/baseline/"
thres="10 12 14 16 18 20 25 30 35 40"
thres="0.1,0.2,0.3,0.4,0.5,0.6"
true_uttids="data/test_50_wrong_trans/true_uttids"
false_uttids="data/test_50_wrong_trans/false_uttids"

# Scoring opts
remove=""
phone_map="data/lang/phones_48_to_39.txt"
duplicate="no"

utts=`cat $uttids`
perfile=$resultdir/per_sil_39_no_dup.txt
[ -f $perfile ] && rm $perfile
cat $uttids | python utils/score_parallel.py \
    --phone_map $phone_map \
    --duplicate $duplicate \
    $trans $hyp >> $perfile || exit 1;

python utils/compute-det.py --thres $thres \
    $true_uttids \
    $false_uttids \
    $perfile \
    $resultdir/fn_fp.txt



