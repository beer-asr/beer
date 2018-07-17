#!/bin/bash


if [ $# -ne 2 ]; then
    echo "$0: setup.sh feature.conf"
    exit 1
fi
setup=$1
feat_conf=$2
stage=-1
. $setup
. $feat_conf

[ ! -d $hmm_model_dir ] && mkdir -p $hmm_model_dir && exit 1;

cp $setup $hmm_model_dir
cp $feat_conf $datadir
cp $feat_conf $hmm_model_dir

if [ $stage -le 0 ]; then
    echo "Accumulating data stastics"
    python3 steps/accumulate_data_stats.py $feats $feat_stats
fi

if [ $stage -le 1 ]; then
    echo "Create emission models"
    python3 steps/create_emission.py $nstates $feat_stats $hmm_model_dir \
        --emission_type $emission_type \
        --noise_std $noise_std || exit 1
fi

echo "Train hmm models"
#qsub -l "gpu=1,hostname=c*,mem_free=2G,ram_free=2G" \
#    -sync y -cwd -j y -o q.log -v setup=$setup\
    steps/train_hmm_cmd.sh $setup
