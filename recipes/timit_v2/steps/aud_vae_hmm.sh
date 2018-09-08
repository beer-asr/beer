#!/bin/bash

# Acoustic Unit Discvoery.


if [ $# -ne 5 ];then
    echo "$0: <setup.sh> <lang-dir> <init-ali> <data-train-dir> <mdl-dir>"
    exit 1
fi

setup=$1
lang_dir=$2
init_ali=$3
data_train_dir=$4
mdl_dir=$5
mkdir -p $mdl_dir

. $setup

fea=$data_train_dir/${aud_vae_hmm_fea_type}.npz
feastats=$data_train_dir/${aud_vae_hmm_fea_type}.stats.npz


if [ ! -f $mdl_dir/0.mdl ]; then
    echo "Building the VAE-HMM AUD model..."

    cat $aud_vae_hmm_hmm_conf | sed s/{n_units}/$aud_vae_hmm_n_units/g \
        > $mdl_dir/hmm.yml || exit 1

    # Creete the unigral LM of the acoustic units.
    python utils/lm-unigram-create.py \
        --concentration $aud_vae_hmm_lm_concentration \
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

    # Get the dimension of the data.
    pycmd="import numpy as np; \
    utts = np.load('$fea'); \
    print(utts[utts.files[0]].shape[1])"
    feadim=$(python -c "$pycmd")

    # Create the phones' hmm graph and their respective emissions.
    python utils/hmm-create-graph-and-emissions.py \
        --dim $aud_vae_hmm_latent_dim \
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
        $mdl_dir/latent_model.mdl || exit 1

    # Create the VAE.
    vars="dim_in=$feadim,dim_out=$aud_vae_hmm_nnet_width"
    python utils/nnet-create.py \
        --set "$vars" \
        $aud_vae_hmm_encoder_conf \
        $mdl_dir/encoder.mdl || exit_msg "Failed to create the VAE encoder"

    vars="dim_in=$aud_vae_hmm_latent_dim,width=$aud_vae_hmm_nnet_width,dim_out=$feadim"
    python utils/nnet-create.py \
        --set "$vars" \
        $aud_vae_hmm_decoder_conf \
        $mdl_dir/decoder.mdl || exit_msg "Failed to create the VAE decoder"

    vars="dim_in=$aud_vae_hmm_latent_dim"
    python utils/nflow-create.py \
        --set "$vars" \
        $aud_vae_hmm_nflow_conf \
        $mdl_dir/nflow.mdl || exit_msg "Failed to create the VAE norm. flow"

    python utils/vae-hmm-create.py \
        --encoder-cov-type $aud_vae_hmm_encoder_cov_type \
        --decoder-cov-type $aud_vae_hmm_decoder_cov_type \
        $feastats \
        $aud_vae_hmm_nnet_width \
        $aud_vae_hmm_latent_dim \
        $mdl_dir/encoder.mdl \
        $mdl_dir/nflow.mdl \
        $mdl_dir/latent_model.mdl \
        $mdl_dir/decoder.mdl \
        $mdl_dir/0.mdl || exit_msg "Failed to create the VAE"
else
    echo "Using previous created HMM: $mdl_dir/0.mdl"
fi


# Make sure all temporary directories will be cleaned up whatever
# happens.
trap "rm -fr $mdl_dir/beer.tmp* &" EXIT


# Train the model.
if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training HMM-GMM model"

    # Retrieve the last model.
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    iter="${mdl%.*}"

    if [ $iter -ge 1 ]; then
        echo "Found existing model. Starting from iteration: $((iter + 1))"
    else
        # Copy the provided alignments  to initalize the system.
        cp $init_ali $mdl_dir/alis.npz
    fi

    while [ $((++iter)) -le $aud_vae_hmm_train_iters ]; do
        echo "Iteration: $iter"

        if echo $aud_vae_hmm_align_iters | grep -w $iter >/dev/null; then
            echo "Aligning data"

            tmpdir=$(mktemp -d $mdl_dir/beer.tmp.XXXX);
            cmd="python utils/vae-hmm-align.py \
                $mdl_dir/$mdl  $fea $tmpdir"
            utils/parallel/submit_parallel.sh \
                "$parallel_env" \
                "vae-hmm-align-iter$iter" \
                "$aud_vae_hmm_align_parallel_opts" \
                "$aud_vae_hmm_align_njobs" \
                "$data_train_dir/uttids" \
                "$cmd" \
                $mdl_dir || exit 1
            find $tmpdir -name '*npy' | \
                  zip -j -@ $mdl_dir/alis.npz > /dev/null || exit 1
            cp $mdl_dir/alis.npz $mdl_dir/alis.${iter}.npz

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
            python utils/vae-hmm-set-decoding-graph.py \
                $mdl_dir/$mdl \
                $mdl_dir/decode_graph.pkl \
                $mdl_dir/phones_hmm.graphs \
                $mdl_dir/$mdl || exit
        fi

        # Clean up the tmp directory to avoid the disk usage to
        # grow without bound.
        rm -fr $tmpdir &

        kl_div_weight="--kl-weight 1.0"
        if [ $iter -le $aud_vae_hmm_train_warmup_iters ]; then
            echo "pre-training (KL div. weight is 0.)"
            kl_div_weight="--kl-weight 0.0"
        fi

        echo "Training the emissions"
        cmd="python -u utils/vae-hmm-train-with-alignments.py \
                --verbose \
                --batch-size $aud_vae_hmm_train_batch_size \
                --lrate $aud_vae_hmm_train_lrate \
                --lrate-nnet $aud_vae_hmm_train_nnet_lrate \
                --epochs $aud_vae_hmm_train_epochs_per_iter \
                --nnet-optim-state $mdl_dir/nnet_optim_state.pkl \
                $kl_div_weight \
                $aud_vae_hmm_train_opts \
                $mdl_dir/$((iter - 1)).mdl \
                $mdl_dir/alis.npz \
                $fea \
                $feastats \
                $mdl_dir/${iter}.mdl"
        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "vae-hmm-train-iter$iter" \
            "$aud_vae_hmm_train_parallel_opts" \
            "$cmd" \
            $mdl_dir || exit 1

        mdl=${iter}.mdl
    done

    ln -s $mdl_dir/$mdl $mdl_dir/final.mdl
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

