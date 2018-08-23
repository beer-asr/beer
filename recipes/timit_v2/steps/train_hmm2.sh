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

    rm -fr $mdl_dir/log/
    mkdir -p $mdl_dir/log

    # Split the utterances into X batches to parallelize the
    # alignments.
    mkdir -p $mdl_dir/split
    cp $data_train_dir/uttids $mdl_dir/split/uttids
    pushd $mdl_dir/split > /dev/null
    split --numeric-suffixes=1 -n l/$hmm_ali_njobs ./uttids
    popd > /dev/null

    # Retrieve the last model (ordered by number).
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    align_no="${mdl%.*}"

    while [ $align_no -le $hmm_n_align ]; do
        tmpdir=$(mktemp -d $mdl_dir/tmp.XXXX);
        trap 'rm -rf "$tmpdir"' EXIT

        echo "Iteration: $align_no"

        if [ ! -f $mdl_dir/alis.$align_no.npz ]; then
            echo "Aligning the features to the transcription"
            cmd="python utils/hmm-align.py \
                --utt-graphs $mdl_dir/ali_graphs.npz \
                $mdl_dir/$mdl \
                $data_train_dir/feats.npz \
                $tmpdir"
            qsub \
                -N "beer-align" \
                -cwd \
                -j y \
                -o $mdl_dir/log/align.$align_no.out.'$TASK_ID' \
                -t 1-$hmm_ali_njobs \
                -sync y \
                $fea_sge_opts \
                utils/jobarray.qsub "$cmd" $mdl_dir/split || exit 1
            find $tmpdir -name '*npy' | \
                zip -j -@ $mdl_dir/alis.$align_no.npz > /dev/null || exit 1
            rm -fr $tmpdir
        else
            echo "Using alignments: $mdl_dir/alis.$align_no.npz"
        fi

        echo "Training the emissions"
        cmd="python -u utils/hmm-train-with-alignments.py \
                --batch-size $hmm_train_emissions_batch_size \
                --epochs $hmm_train_emissions_epochs \
                --lrate $hmm_train_emissions_lrate \
                $hmm_train_emissions_opts \
                $mdl_dir/$mdl \
                $mdl_dir/alis.$align_no.npz \
                $data_train_dir/feats.npz \
                $data_train_dir/feats.stats.npz \
                $mdl_dir/$((align_no + 1)).mdl"
        qsub \
            -N "beer-train-emissions" \
            -cwd -j y \
            -o $mdl_dir/log/train_emissions.$align_no.out \
            -sync y \
            $hmm_train_emissions_sge_opts \
            utils/job.qsub "$cmd" || exit 1

        # Retrieve the last model (ordered by number).
        mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
            sort -t '.' -k 1 -g | tail -1)
        align_no="${mdl%.*}"
    done
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

