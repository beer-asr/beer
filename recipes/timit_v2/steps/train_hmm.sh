#!/bin/bash

exit_msg (){
    echo "$1"
    exit 1
}

if [ $# -ne 3 ];then
    echo "$0: <setup.sh> <data-train-dir> <mdl-dir>"
    exit 1
fi

setup=$1
data_train_dir=$2
mdl_dir=$3
mkdir -p $mdl_dir/log

. $setup

[ -f "$hmm_emission_conf" ] || exit_msg "File not found: $hmm_emission_conf"

#if [ ! -d $mdl_dir ];then
#    mkdir -p $mdl_dir/log
#    cp $setup $mdl_dir
#    cp $fea_conf $mdl_dir
#    cp $hmm_emission_conf $mdl_dir
#fi

if [ ! -f $mdl_dir/init.mdl ]; then
    echo "Building the HMM model..."
    # Build the decoding graph.
    python utils/prepare-decode-graph.py \
        $langdir/phone_graph.txt $mdl_dir/decode_graph || exit 1

    # Create the phones' hmm graph and their respective emissions.
    python utils/hmm-create-graph-and-emissions.py \
        --stats $data_train_dir/feats.stats.npz \
         $hmm_emission_conf \
         $langdir/phones.txt \
         $mdl_dir/phones_hmm \
         $mdl_dir/emissions || exit 1

    # Create the HMM model.
    python utils/hmm-create.py \
        $mdl_dir/decode_graph \
        $mdl_dir/phones_hmm \
        $mdl_dir/emissions \
        $mdl_dir/init.mdl || exit 1
else
    echo "Using previous created HMM: $mdl_dir/init.mdl"
fi


if [ ! -f $mdl_dir/ali_graph.npz ]; then
    echo "Preparing the alignment graph..."

    tmpdir=$(mktemp -d /tmp/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT

    python utils/prepare-alignments.py \
        $mdl_dir/phones_hmm \
        $data_train_dir/trans \
        $tmpdir
    find $tmpdir -name '*npy' \
        | zip -j -@ $mdl_dir/ali_graph.npz > /dev/null || exit 1
else
    echo "Alignment already prepared: $mdl_dir/ali.npz"
fi

if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training HMM-GMM model"
    rm -fr $mdl_dir/log/sge.log
    cmd="python -u -m cProfile -s cumtime utils/train_hmm.py \
        --infer-type $hmm_infer_type \
        --lrate $hmm_lrate \
        --batch-size $hmm_batch_size \
        --epochs $hmm_epochs \
        $use_gpu $hmm_fast_eval \
        $data_train_dir/feats.npz \
        $mdl_dir/ali_graph.npz \
        $mdl_dir/init.mdl \
        $data_train_dir/feats.stats.npz \
        $mdl_dir"
    qsub -N "beer-hmm-gmm" -cwd -j y -o $mdl_dir/log/sge.log -sync y \
        -l gpu=1,hostname="b1[1-9]*|c*" \
        utils/job.qsub "$cmd" || exit 1
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi
