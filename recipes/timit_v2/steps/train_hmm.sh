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

# Copy the configuration files for information.
if [ ! -d $mdl_dir ];then
    mkdir -p $mdl_dir/log
fi
cp $setup $mdl_dir/setup.sh
cp $fea_conf $mdl_dir/feats.conf
cp $hmm_emission_conf $mdl_dir/emissions.yml

if [ ! -f $mdl_dir/init.mdl ]; then
    echo "Building the HMM model..."
    # Build the decoding graph.
    python utils/prepare-decode-graph.py \
        $langdir/phone_graph.txt $mdl_dir/decode.graph || exit 1

    # Create the phones' hmm graph and their respective emissions.
    python utils/hmm-create-graph-and-emissions.py \
        --stats $data_train_dir/feats.stats.npz \
         $hmm_emission_conf \
         $langdir/phones.txt \
         $mdl_dir/phones_hmm.graphs \
         $mdl_dir/emissions.mdl || exit 1

    # Create the pdf to phone mapping (used for decoding).
    python utils/hmm-pdf-id-mapping.py $mdl_dir/phones_hmm.graphs \
        > $mdl_dir/pdf_mapping.txt

    # Create the HMM model.
    python utils/hmm-create.py \
        $mdl_dir/decode.graph \
        $mdl_dir/phones_hmm.graphs \
        $mdl_dir/emissions.mdl \
        $mdl_dir/0.mdl || exit 1
else
    echo "Using previous created HMM: $mdl_dir/init.mdl"
fi


# Prepare the alignments.
if [ ! -f $mdl_dir/ali_graphs.npz ]; then
    echo "Preparing the alignment graph..."

    tmpdir=$(mktemp -d /tmp/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT

    python utils/prepare-alignments.py \
        $mdl_dir/phones_hmm.graphs \
        $data_train_dir/trans \
        $tmpdir
    find $tmpdir -name '*npy' \
        | zip -j -@ $mdl_dir/ali_graphs.npz > /dev/null || exit 1
else
    echo "Alignment already prepared: $mdl_dir/ali.npz"
fi

if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training HMM-GMM model"

    # Retrieve the last model (ordered by number) and use it as
    # initial model for the training.
    start_mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)

    rm -fr $mdl_dir/log/sge.log

    cmd="python -u utils/hmm-train.py \
            --alignments $mdl_dir/ali_graphs.npz \
            --batch-size $hmm_batch_size \
            --epochs $hmm_epochs \
            --infer-type $hmm_infer_type \
            --lrate $hmm_lrate \
            --tmpdir $mdl_dir \
            $use_gpu $hmm_fast_eval \
            $mdl_dir/$start_mdl \
            $data_train_dir/feats.npz \
            $data_train_dir/feats.stats.npz \
            $mdl_dir/final.mdl"
    qsub \
        -N "beer-hmm-gmm" \
        -cwd -j y \
        -o $mdl_dir/log/sge.log -sync y \
        $hmm_train_sge_opts \
        utils/job.qsub "$cmd" || exit 1
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

