#!/bin/bash

# Train a context independent HMM based Phone Recognizer.


if [ $# -ne 4 ];then
    echo "$0: <setup.sh> <init-ali> <data-train-dir> <mdl-dir>"
    exit 1
fi

setup=$1
init_ali=$2
data_train_dir=$3
mdl_dir=$4
mkdir -p $mdl_dir


. $setup
[[ -f "$vae_hmm_hmm_conf" ]] || \
    { echo "File not found: $vae_hmm_hmm_conf"; exit 1; }
[[ -f "$vae_hmm_encoder_conf" ]] || \
    { echo "File not found: $vae_hmm_encoder_conf"; exit 1; }
[[ -f "$vae_hmm_decoder_conf" ]] || \
    { echo "File not found: $vae_hmm_decoder_conf"; exit 1; }
[[ -f "$vae_hmm_nflow_conf" ]] || \
    { echo "File not found: $vae_hmm_nflow_conf"; exit 1; }


if [ ! -f $mdl_dir/0.mdl ]; then
    echo "Building the VAE-HMM model..."

    # Copy the configuration files for information.
    cp $setup $mdl_dir
    cp $fea_conf $mdl_dir
    cp $vae_hmm_hmm_conf $mdl_dir
    cp $vae_hmm_encoder_conf $mdl_dir
    cp $vae_hmm_decoder_conf $mdl_dir
    cp $vae_hmm_nflow_conf $mdl_dir

    # Get the dimension of the data.
    pycmd="import numpy as np; \
    utts = np.load('$data_train_dir/feats.npz'); \
    print(utts[utts.files[0]].shape[1])"
    feadim=$(python -c "$pycmd")

    # Build the decoding graph.
    python utils/prepare-decode-graph.py \
        $langdir/phone_graph.txt $mdl_dir/decode.graph || exit 1

    # Create the phones' hmm graph and their respective emissions.
    python utils/hmm-create-graph-and-emissions.py \
        --dim $vae_hmm_latent_dim \
         $vae_hmm_hmm_conf \
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
        $mdl_dir/latent_model.mdl || exit 1

    # Create the VAE.
    vars="dim_in=$feadim,dim_out=$vae_hmm_nnet_width"
    python utils/nnet-create.py \
        --set "$vars" \
        $vae_hmm_encoder_conf \
        $mdl_dir/encoder.mdl || exit_msg "Failed to create the VAE encoder"

    vars="dim_in=$vae_hmm_latent_dim,width=$vae_hmm_nnet_width,dim_out=$feadim"
    python utils/nnet-create.py \
        --set "$vars" \
        $vae_hmm_decoder_conf \
        $mdl_dir/decoder.mdl || exit_msg "Failed to create the VAE decoder"

    vars="dim_in=$vae_hmm_latent_dim"
    python utils/nflow-create.py \
        --set "$vars" \
        $vae_hmm_nflow_conf \
        $mdl_dir/nflow.mdl || exit_msg "Failed to create the VAE norm. flow"

    python utils/vae-hmm-create.py \
        --encoder-cov-type $vae_hmm_encoder_cov_type \
        --decoder-cov-type $vae_hmm_decoder_cov_type \
        $data_train_dir/feats.stats.npz \
        $vae_hmm_nnet_width \
        $vae_hmm_latent_dim \
        $mdl_dir/encoder.mdl \
        $mdl_dir/nflow.mdl \
        $mdl_dir/latent_model.mdl \
        $mdl_dir/decoder.mdl \
        $mdl_dir/0.mdl || { echo "Failed to create the VAE" ; exit 1; }
else
    echo "Using previous created VAE-HMM: $mdl_dir/0.mdl"
fi


# Make sure all temporary directories will be cleaned up whatever
# happens.
trap "rm -rf $mdl_dir/beer.tmp* &" EXIT


# Prepare the alignments the alignemnts graphs.
if [ ! -f $mdl_dir/ali_graphs.npz ]; then
    echo "Preparing alignment graphs..."

    tmpdir=$(mktemp -d $mdl_dir/beer.tmp.XXXX);

    cmd="python utils/prepare-alignments.py \
            $mdl_dir/phones_hmm.graphs $tmpdir"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "prepare-align" \
        "$vae_hmm_align_parallel_opts" \
        "$vae_hmm_align_njobs" \
        "$data_train_dir/trans" \
        "$cmd" \
        $mdl_dir || exit 1
    find $tmpdir -name '*npy' \
        | zip -j -@ $mdl_dir/ali_graphs.npz > /dev/null || exit 1

else
    echo "Alignment graphs already prepared: $mdl_dir/ali_graphs.npz"
fi


if [ ! -f $mdl_dir/alis.npz ]; then
    echo "Using: $init_ali as initial alignments."
    ln -s $init_ali $mdl_dir/alis.npz
fi


# Train the model.
if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training the VAE-HMM model"

    # Retrieve the last model.
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    iter="${mdl%.*}"

    while [ $((++iter)) -le $vae_hmm_train_iters ]; do
        echo "Iteration: $iter"

        if echo $vae_hmm_align_iters | grep -w $iter >/dev/null; then
            echo "Aligning data"

            tmpdir=$(mktemp -d $mdl_dir/tmp.XXXX);
            cmd="python utils/vae-hmm-align.py \
                --ali-graphs $mdl_dir/ali_graphs.npz \
                $mdl_dir/$mdl  $data_train_dir/feats.npz  $tmpdir"
            utils/parallel/submit_parallel.sh \
                "$parallel_env" \
                "vae-align" \
                "$vae_hmm_align_parallel_opts" \
                "$vae_hmm_align_njobs" \
                "$data_train_dir/uttids" \
                "$cmd" \
                $mdl_dir/iter$iter || exit 1
            find $tmpdir -name '*npy' | \
                  zip -j -@ $mdl_dir/alis.npz > /dev/null || exit 1
        fi

        # Clean up the tmp directory to avoid the disk usage to
        # grow without bound.
        rm -fr $tmpdir &

        kl_div_weight="--kl-weight 1.0"
        if [ $iter -le $vae_hmm_train_warmup_iters ]; then
            echo "pre-training (KL div. weight is 0.)"
            kl_div_weight="--kl-weight 0.0"
        fi

        echo "Training the emissions"
        cmd="python -u utils/vae-hmm-train-with-alignments.py \
                --verbose \
                --batch-size $vae_hmm_train_batch_size \
                --lrate $vae_hmm_train_lrate \
                --epochs $vae_hmm_train_epochs_per_iter \
                --nnet-optim-state $mdl_dir/nnet_optim_state.pkl \
                $kl_div_weight \
                $vae_hmm_train_opts \
                $mdl_dir/$((iter - 1)).mdl \
                $mdl_dir/alis.npz \
                $data_train_dir/feats.npz \
                $data_train_dir/feats.stats.npz \
                $mdl_dir/${iter}.mdl"
        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "vae-hmm-train" \
            "$vae_hmm_train_parallel_opts" \
            "$cmd" \
            $mdl_dir/iter$iter || exit 1

        mdl=${iter}.mdl
    done

    cp $mdl_dir/$mdl $mdl_dir/final.mdl
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

