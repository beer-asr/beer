#!/bin/bash

# Train a context independent HMM based Phone Recognizer.


if [ $# -ne 3 ];then
    echo "$0: <setup.sh> <data-train-dir> <mdl-dir>"
    exit 1
fi

setup=$1
data_train_dir=$2
mdl_dir=$3
mkdir -p $mdl_dir


. $setup
[[ -f "$hmm_emission_conf" ]] || \
    { echo "File not found: $hmm_emission_conf"; exit 1; }



if [ ! -f $mdl_dir/0.mdl ]; then
    echo "Building the HMM model..."

    # Copy the configuration files for information.
    cp $setup $mdl_dir
    cp $fea_conf $mdl_dir
    cp $hmm_emission_conf $mdl_dir

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


# Make sure all temporary directories will be cleaned up whatever
# happens.
trap 'rm -rf "beer.tmp*"' EXIT


# Prepare the alignments the alignemnts graphs.
if [ ! -f $mdl_dir/ali_graphs.npz ]; then
    echo "Preparing alignment graphs..."

    tmpdir=$(mktemp -d $mdl_dir/beer.tmp.XXXX);

    cmd="python utils/prepare-alignments.py \
            $mdl_dir/phones_hmm.graphs $tmpdir"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "prepare-align" \
        "$hmm_align_parallel_opts" \
        "$hmm_align_njobs" \
        "$data_train_dir/trans" \
        "$cmd" \
        $mdl_dir || exit 1

    find $tmpdir -name '*npy' \
        | zip -j -@ $mdl_dir/ali_graphs.npz > /dev/null || exit 1

else
    echo "Alignment graphs already prepared: $mdl_dir/ali_graphs.npz"
fi


# Train the model.
if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training HMM-GMM model"

    # Retrieve the last model.
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    iter="${mdl%.*}"

    if [ $iter -ge 1 ]; then
        echo "Found existing model. Starting from iteration: $((iter + 1))"
    fi

    while [ $((++iter)) -le $hmm_train_iters ]; do
        echo "Iteration: $iter"

        if echo $hmm_align_iters | grep -w $iter >/dev/null; then
            echo "Aligning data"

            tmpdir=$(mktemp -d $mdl_dir/tmp.XXXX);
            cmd="python utils/hmm-align.py \
                --ali-graphs $mdl_dir/ali_graphs.npz \
                $mdl_dir/$mdl  $data_train_dir/feats.npz  $tmpdir"
            utils/parallel/submit_parallel.sh \
                "$parallel_env" \
                "align" \
                "$hmm_align_parallel_opts" \
                "$hmm_align_njobs" \
                "$data_train_dir/uttids" \
                "$cmd" \
                $mdl_dir || exit 1
            find $tmpdir -name '*npy' | \
                  zip -j -@ $mdl_dir/alis.npz > /dev/null || exit 1
        fi

        # Clean up the tmp directory to avoid the disk usage to
        # grow without bound.
        rm -fr $tmpdir &

        echo "Training the emissions"
        cmd="python -u utils/hmm-train-with-alignments.py \
                --batch-size $hmm_train_emissions_batch_size \
                --lrate $hmm_train_emissions_lrate \
                $hmm_train_emissions_opts \
                $mdl_dir/$((iter - 1)).mdl \
                $mdl_dir/alis.npz \
                $data_train_dir/feats.npz \
                $data_train_dir/feats.stats.npz \
                $mdl_dir/${iter}.mdl"
        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "hmm-train" \
            "$hmm_train_parallel_opts" \
            "$cmd" \
            $mdl_dir || exit 1

        mdl=${iter}.mdl
    done

    ln -s $mdl_dir/$mdl $mdl_dir/final.mdl
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

