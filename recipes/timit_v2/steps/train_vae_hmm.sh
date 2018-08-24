#!/bin/bash


# Train a mono-phone VAE-HMM model.


exit_msg () {
    echo "$1"
    exit 1
}


if [ $# -ne 3 ]; then
    echo "Usage: $0 <setup.sh> <train-data-dir> <out-model-dir>"
    exit 1
fi

# Load the settings
setup=$1
. $setup

data_train_dir=$2
mdl_dir=$3

encoder_conf=$vae_hmm_encoder_conf
decoder_conf=$vae_hmm_decoder_conf
emissions_conf=$vae_hmm_emissions_conf
nflow_conf=$vae_hmm_normalizing_flow_conf
features=$data_train_dir/feats.npz

# Check if all the inputs are here.
for f in $encoder_conf $decoder_conf $emissions_conf $nflow_conf $features; do
    [ -f $f ] || exit_msg "File not found: $f"
done

if [ ! -d $mdl_dir ];then
    mkdir -p $mdl_dir/log
    for f in $setup $fea_conf $encoder_conf $decoder_conf $emissions_conf $nflow_conf; do
        cp $f $mdl_dir || exit_msg "File copy failed: $f"
    done
fi

# Get the dimension of the data.
pycmd="import numpy as np; \
utts = np.load('$features'); \
print(utts[utts.files[0]].shape[1])"
feadim=$(python -c "$pycmd")

if [ ! -f $mdl_dir/states.int.npz ]; then
    echo "Convert the transcription to the state sequence."
    python utils/prepare_state_labels.py \
        $langdir/phones.txt \
        $data_train_dir/phones.int.npz \
        $vae_hmm_emissions_conf \
        $mdl_dir || exit_msg "Failed to create the state labels"
else
    echo "Phone state labels already created in $mdl_dir/states.int.npz"
fi

#######################################################################
# Model creation.

if [ ! -f $mdl_dir/0.mdl ]; then
    echo "Creating the VAE-HMM model..."

    mkdir -p "$mdl_dir"/init

    # Create the components of the VAE.
    vars="dim_in=$feadim,dim_out=$vae_hmm_encoder_out_dim"
    python utils/nnet-create.py \
        --set "$vars" \
        $encoder_conf \
        $mdl_dir/init/encoder.mdl || exit_msg "Failed to create the VAE encoder"

    vars="latent_dim=$vae_hmm_latent_dim, \
    encoder_out_dim=$vae_hmm_encoder_out_dim, \
    dim_out=$feadim"
    python utils/nnet-create.py \
        --set "$vars" \
        $decoder_conf \
        $mdl_dir/init/decoder.mdl || exit_msg "Failed to create the VAE decoder"

    vars="dim_in=$vae_hmm_latent_dim"
    python utils/nflow-create.py \
        --set "$vars" \
        $nflow_conf \
        $mdl_dir/init/nflow.mdl || exit_msg "Failed to create the VAE norm. flow"

    python utils/create_emission.py \
        --dim $vae_hmm_latent_dim \
        $vae_hmm_emissions_conf \
        $mdl_dir/init/emissions.mdl || exit_msg "Failed to create the emissions"

    python utils/create_vae.py \
        --encoder-cov-type $vae_hmm_encoder_cov_type \
        --decoder-cov-type $vae_hmm_decoder_cov_type \
        $data_train_dir/feats.stats.npz \
        $vae_hmm_encoder_out_dim \
        $vae_hmm_latent_dim \
        $mdl_dir/init/encoder.mdl \
        $mdl_dir/init/nflow.mdl \
        $mdl_dir/init/emissions.mdl \
        $mdl_dir/init/decoder.mdl \
        $mdl_dir/0.mdl || exit_msg "Failed to create the VAE"
else
    echo "Initial model already created: $mdl_dir/0.mdl"
fi

if [ ! -f $mdl_dir/final.mdl ]; then
    echo "Training..."

    python -u utils/train_vae_hmm.py \
        --training_type $vae_hmm_training_type \
        --lrate $vae_hmm_lrate \
        --lrate-nnet $vae_hmm_lrate_nnet \
        --batch-size $vae_hmm_batch_size \
        --epochs $vae_hmm_epochs \
        $vae_hmm_opts \
        $data_train_dir/feats.npz \
        $mdl_dir/states.int.npz \
        $mdl_dir/0.mdl \
        $data_train_dir/feats.stats.npz \
        $mdl_dir > $mdl_dir/log/vae-hmm-train.log 2>&1 || exit_msg "Training failed"
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

