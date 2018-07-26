#!/bin/bash


if [ $# -ne 2 ]; then
    echo "$0: setup.sh feature.conf"
    exit 1
fi
setup=$1
feat_conf=$2
stage=1
. $setup
. $feat_conf

if [ ! -d $hmm_model_dir ]; then
    mkdir -p $hmm_model_dir || exit "Failed to mkdir $hmm_model_dir";
else
    rm -r $hmm_model_dir/*.log
    rm -r $hmm_model_dir/*.sh
fi


echo $setup
cp $setup $hmm_model_dir || exit "Cannot copy $setup to $hmm_model_dir"
cp $feat_conf $datadir 
cp $feat_conf $hmm_model_dir || exit "Cannot copy $feat_conf to
    $hmm_model_dir"
cp $emiss_conf $hmm_model_dir || exit "Cannot copy $emiss_conf to
$hmm_model_dir"

if [ $stage -le 0 ]; then
    echo "Accumulating data stastics"
    python3 steps/accumulate_data_stats.py $feats $feat_stats
fi

if [ $stage -le 1 ]; then
    echo "Create emission models"
    python3 steps/create_emission.py $emiss_conf $feat_stats $hmm_model_dir || exit 1
fi

echo "Train hmm models"
if [ -z $use_gpu ]; then
    echo "Training on CPUs"
    steps/train_hmm_cmd.sh $setup > $hmm_model_dir/train.log 2>&1
else
    echo "Training on GPUs"
qsub -l "gpu=1,hostname=b1[12345678]*|c*,mem_free=2G,ram_free=2G" \
    -sync y -cwd -j y -o $hmm_model_dir/q.log -v setup=$setup\
    steps/train_hmm_cmd.sh $setup > $hmm_model_dir/train.log 2>&1
fi
