#!/bin/bash

# Acoustic Unit Discvoery.


if [ $# -ne 4 ];then
    echo "$0: <setup.sh> <lang-dir> <data-train-dir> <mdl-dir>"
    exit 1
fi

setup=$1
lang_dir=$2
data_train_dir=$3
mdl_dir=$4
mkdir -p $mdl_dir


. $setup

fea=$data_train_dir/${aud_vae_hmm_fea_type}.npz
feastats=$data_train_dir/${aud_vae_hmm_fea_type}.stats.npz

[[ -f "$aud_hmm_conf" ]] || \
    { echo "File not found: $aud_hmm_conf"; exit 1; }



if [ ! -f $mdl_dir/0.mdl ]; then
    echo "Building the HMM model..."

    cat $aud_hmm_conf | sed s/{n_units}/$aud_hmm_n_units/g \
        > $mdl_dir/hmm.yml || exit 1

    # Creete the unigral LM of the acoustic units.
    python utils/lm-unigram-create.py \
        --concentration $aud_hmm_lm_concentration \
        $(cat $lang_dir/phones.txt | wc -l) \
        $mdl_dir/lm.mdl || exit 1

    # Create the definition of the decoding graph.
    python utils/create-decode-graph.py \
       --unigram-lm $mdl_dir/lm.mdl \
       --use-silence \
       $lang_dir/phones.txt > $mdl_dir/decode_graph.txt || exit 1

    # Build the decoding graph.
    python utils/prepare-decode-graph.py \
        $mdl_dir/decode_graph.txt $mdl_dir/decode_graph.pkl || exit 1

    # Create the phones' hmm graph and their respective emissions.
    python utils/hmm-create-graph-and-emissions.py \
        --stats $feastats \
         $mdl_dir/hmm.yml  \
         $lang_dir/phones.txt \
         $mdl_dir/phones_hmm.graphs \
         $mdl_dir/emissions.mdl || exit 1

    # Create the pdf to phone mapping (used for decoding).
    python utils/hmm-pdf-id-mapping.py $mdl_dir/phones_hmm.graphs \
        > $mdl_dir/pdf_mapping.txt

    # Create the HMM model.
    python utils/hmm-create.py \
        $mdl_dir/decode_graph.pkl \
        $mdl_dir/phones_hmm.graphs \
        $mdl_dir/emissions.mdl \
        $mdl_dir/0.mdl || exit 1
else
    echo "Using previous created HMM: $mdl_dir/0.mdl"
fi


# Make sure all temporary directories will be cleaned up whatever
# happens.
trap "rm -fr $mdl_dir/beer.tmp* &" EXIT


# Train the model.
if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training HMM model"

    # Retrieve the last model.
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    iter="${mdl%.*}"

    if [ $iter -ge 1 ]; then
        echo "Found existing model. Starting from iteration: $((iter + 1))"
    fi

    while [ $((++iter)) -le $hmm_train_iters ]; do
        echo "Iteration: $iter"

        if echo $aud_hmm_align_iters | grep -w $iter >/dev/null; then
            echo "Aligning data"

            tmpdir=$(mktemp -d $mdl_dir/beer.tmp.XXXX);
            cmd="python utils/hmm-align.py \
                $mdl_dir/$mdl $fea $tmpdir"
            utils/parallel/submit_parallel.sh \
                "$parallel_env" \
                "hmm-align-iter$iter" \
                "$aud_hmm_align_parallel_opts" \
                "$aud_hmm_align_njobs" \
                "$data_train_dir/uttids" \
                "$cmd" \
                $mdl_dir || exit 1
            find $tmpdir -name '*npy' | \
                  zip -j -@ $mdl_dir/alis.npz > /dev/null || exit 1

            echo "Re-estimating the unigram LM of the units..."

            # Convert the alignments into units best paths.
            tmpdir=$(mktemp -d $mdl_dir/beer.tmp.XXXX);
            python utils/convert-ali-to-best-path.py \
                $mdl_dir/alis.npz \
                $mdl_dir/pdf_mapping.txt \
                $lang_dir/phones.txt \
                $tmpdir || exit 1
            find $tmpdir -name '*npy' | \
                  zip -j -@ $mdl_dir/best_paths.npz > /dev/null || exit 1

            # Re-estimate the language model.
            python utils/lm-unigram-reestimate.py \
                $mdl_dir/lm.mdl \
                $mdl_dir/best_paths.npz \
                $mdl_dir/lm.mdl || exit 1

            # Create the definition of the decoding graph.
            python utils/create-decode-graph.py \
               --unigram-lm $mdl_dir/lm.mdl \
               --use-silence \
               $lang_dir/phones.txt > $mdl_dir/decode_graph.txt || exit 1

            # Build the decoding graph.
            python utils/prepare-decode-graph.py \
                $mdl_dir/decode_graph.txt $mdl_dir/decode_graph.pkl || exit 1

            # Update the HMM model.
            python utils/hmm-set-decoding-graph.py \
                $mdl_dir/$mdl \
                $mdl_dir/decode_graph.pkl \
                $mdl_dir/phones_hmm.graphs \
                $mdl_dir/$mdl || exit
        fi

        # Clean up the tmp directory to avoid the disk usage to
        # grow without bound.
        rm -fr $tmpdir &

        echo "Training the emissions"
        cmd="python -u utils/hmm-train-with-alignments.py \
                --batch-size $aud_hmm_train_batch_size \
                --lrate $aud_hmm_train_lrate \
                $aud_hmm_train_opts \
                $mdl_dir/$((iter - 1)).mdl \
                $mdl_dir/alis.npz \
                $fea \
                $feastats \
                $mdl_dir/${iter}.mdl"
        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "hmm-train-iter$iter" \
            "$aud_hmm_train_parallel_opts" \
            "$cmd" \
            $mdl_dir || exit 1

        mdl=${iter}.mdl
    done

    ln -s $mdl_dir/$mdl $mdl_dir/final.mdl
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

