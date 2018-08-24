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
    echo "Building the VAE-HMM model..."
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

    python utils/create_vae.py \
        --encoder-cov-type $vae_hmm_encoder_cov_type \
        --decoder-cov-type $vae_hmm_decoder_cov_type \
        $data_train_dir/feats.stats.npz \
        $vae_hmm_nnet_width \
        $vae_hmm_latent_dim \
        $mdl_dir/encoder.mdl \
        $mdl_dir/nflow.mdl \
        $mdl_dir/latent_model.mdl \
        $mdl_dir/decoder.mdl \
        $mdl_dir/0.mdl || exit_msg "Failed to create the VAE"
else
    echo "Using previous created HMM: $mdl_dir/0.mdl"
fi


# Prepare the alignments.
if [ ! -f $mdl_dir/ali_graphs.npz ]; then
    echo "Preparing alignment graphs..."

    tmpdir1=$(mktemp -d /tmp/beer.XXXX);
    python utils/prepare-alignments.py \
        $mdl_dir/phones_hmm.graphs \
        $data_train_dir/trans \
        $tmpdir1
    find $tmpdir1 -name '*npy' \
        | zip -j -@ $mdl_dir/ali_graphs.npz > /dev/null || exit 1
    rm -fr $tmpdir1
else
    echo "Alignment graphs already prepared: $mdl_dir/ali_graphs.npz"
fi

if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training the VAE-HMM model"

    rm -fr $mdl_dir/log/
    mkdir -p $mdl_dir/log

    # Split the utterances into X batches to parallelize the
    # alignments.
    mkdir -p $mdl_dir/split
    cp $data_train_dir/uttids $mdl_dir/split/uttids
    pushd $mdl_dir/split > /dev/null
    split --numeric-suffixes=1 -n l/$vae_hmm_align_njobs ./uttids
    popd > /dev/null

    # Retrieve the last model.
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    epoch="${mdl%.*}"

    while [ $((++epoch)) -le $vae_hmm_train_iters ]; do
        echo "Iteration: $epoch"

        if echo $vae_hmm_align_epochs | grep -w $epoch >/dev/null; then
            echo "Aligning data"

            tmpdir2=$(mktemp -d $mdl_dir/tmp.XXXX);
            cmd="python utils/vae-hmm-align.py \
                --utt-graphs $mdl_dir/ali_graphs.npz \
                $mdl_dir/$mdl  $data_train_dir/feats.npz  $tmpdir2"
            qsub  -N "beer-align"  -cwd  -j y -sync y $vae_hmm_align_sge_opts \
                -o $mdl_dir/log/align.${epoch}.out.'$TASK_ID' \
                -t 1-$vae_hmm_align_njobs \
                utils/jobarray.qsub "$cmd" $mdl_dir/split || exit 1
            find $tmpdir2 -name '*npy' | \
                  zip -j -@ $mdl_dir/alis.npz > /dev/null || exit 1
            rm -fr $tmpdir2
        fi

        kl_div_weight="--kl-weight 1.0"
        if [ $epoch -le $vae_hmm_train_warmup_iters ]; then
            echo "pre-training (KL div. weight is 0.)"
            kl_div_weight="--kl-weight 0.0"
        fi

        echo "Training the emissions"
        cmd="python -u utils/vae-hmm-train-with-alignments.py \
                --verbose \
                --batch-size $vae_hmm_train_emissions_batch_size \
                --lrate $vae_hmm_train_emissions_lrate \
                --epochs $vae_hmm_train_epochs_per_iter \
                $kl_div_weight \
                $vae_hmm_train_emissions_opts \
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
            $vae_hmm_train_emissions_sge_opts \
            utils/job.qsub "$cmd" || exit 1

        mdl=${epoch}.mdl
    done

    cp $mdl_dir/$mdl $mdl_dir/final.mdl
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

