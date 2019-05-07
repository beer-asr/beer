#!/bin/bash

if [ $# != 2 ];then
    echo "$0: <resultdir> <result-file>"
    exit 1
fi

resultdir=$1
wer_file=$2
#wer_file=$resultdir/ali_llhs_var.txt
thres="$(seq -1 0.01 100 | tr '\n' , | sed s'/,$//')"

#thres="$(seq 30 0.1 300 | tr '\n' , | sed s'/,$//')"
#thres="-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
true_uttids=$resultdir/true_uttids
false_uttids=$resultdir/false_uttids

python utils/compute-det.py --thres="$thres" \
    $true_uttids \
    $false_uttids \
    $wer_file \
    $resultdir/det_per.txt

python utils/compute-roc.py --thres="$thres" \
    $true_uttids \
    $false_uttids \
    $wer_file \
    $resultdir/roc_per.txt



