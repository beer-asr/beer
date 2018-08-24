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
    cp $setup $mdl_dir
    cp $fea_conf $mdl_dir
    cp $hmm_emission_conf $mdl_dir
fi

if [ ! -f $mdl_dir/0.mdl ]; then
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
    echo "Using previous created HMM: $mdl_dir/0.mdl"
fi


# Prepare the alignments.
if [ ! -f $mdl_dir/ali_graphs.npz ]; then
    echo "Preparing alignment graphs..."

    tmpdir=$(mktemp -d /tmp/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT

    python utils/prepare-alignments.py \
        $mdl_dir/phones_hmm.graphs \
        $data_train_dir/trans \
        $tmpdir
    find $tmpdir -name '*npy' \
        | zip -j -@ $mdl_dir/ali_graphs.npz > /dev/null || exit 1
else
    echo "Alignment graphs already prepared: $mdl_dir/ali_graphs.npz"
fi


if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training HMM-GMM model"

    rm -fr $mdl_dir/log/
    mkdir -p $mdl_dir/log

    # Split the utterances into X batches to parallelize the
    # alignments.
    mkdir -p $mdl_dir/split
    cp $data_train_dir/uttids $mdl_dir/split/uttids
    pushd $mdl_dir/split > /dev/null
    split --numeric-suffixes=1 -n l/$hmm_align_njobs ./uttids
    popd > /dev/null

    # Retrieve the last model.
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    epoch="${mdl%.*}"

    while [ $((++epoch)) -le $hmm_train_epochs ]; do
        echo "Iteration: $epoch"

        if echo $hmm_align_epochs | grep -w $epoch >/dev/null; then
            echo "Aligning data"

            tmpdir=$(mktemp -d $mdl_dir/tmp.XXXX);
            trap 'rm -rf "$tmpdir"' EXIT

            cmd="python utils/hmm-align.py \
                --utt-graphs $mdl_dir/ali_graphs.npz \
                $mdl_dir/$mdl  $data_train_dir/feats.npz  $tmpdir"
            qsub  -N "beer-align"  -cwd  -j y -sync y $hmm_align_sge_opts \
                -o $mdl_dir/log/align.${epoch}.out.'$TASK_ID' \
                -t 1-$hmm_align_njobs \
                utils/jobarray.qsub "$cmd" $mdl_dir/split || exit 1
            find $tmpdir -name '*npy' | \
                  zip -j -@ $mdl_dir/alis.npz > /dev/null || exit 1
        fi

        echo "Training the emissions"
        cmd="python -u utils/hmm-train-with-alignments.py \
                --batch-size $hmm_train_emissions_batch_size \
                --lrate $hmm_train_emissions_lrate \
                $hmm_train_emissions_opts \
                $mdl_dir/$((epoch - 1)).mdl \
                $mdl_dir/alis.npz \
                $data_train_dir/feats.npz \
                $data_train_dir/feats.stats.npz \
                $mdl_dir/${epoch}.mdl"
        qsub \
            -N "beer-train-emissions" \
            -cwd -j y \
            -o $mdl_dir/log/train_emissions.${epoch}.out \
            -sync y \
            $hmm_train_emissions_sge_opts \
            utils/job.qsub "$cmd" || exit 1

        mdl=${epoch}.mdl
    done

    cp $mdl_dir/$mdl $mdl_dir/final.mdl
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

