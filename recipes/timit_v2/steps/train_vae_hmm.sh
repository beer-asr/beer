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

train_datadir=$2
outdir=$3


encoder_conf=$vae_hmm_encoder_conf
decoder_conf=$vae_hmm_decoder_conf
emissions_conf=$vae_hmm_emissions_conf
nflow_conf=$vae_hmm_normalizing_flow_conf
features=$train_datadir/feats.npz

# Check if all the inputs are here.
[ -f "$encoder_conf" ] || exit_msg "File not found: $encoder_conf"
[ -f "$decoder_conf" ] || exit_msg "File not found: $decoder_conf"
[ -f "$emissions_conf" ] || exit_msg "File not found: $emissions_conf"
[ -f "$nflow_conf" ] || exit_msg "File not found: $nflow_conf"
[ -f $features ] || exit_msg "File not found: $featurs"


# Get the dimension of the data.
pycmd="import numpy as np; \
utts = np.load('$features'); \
print(utts[utts.files[0]].shape[1])"
feadim=$(python -c "$pycmd")

#######################################################################
# Model creation.

if [ ! -f $outdir/model_0.mdl ]; then
    echo "Creating the VAE-HMM model..."

    mkdir -p "$outdir"/init

    # Create the components of the VAE.
    vars="dim_in=$feadim,dim_out=$vae_hmm_encoder_out_dim"
    python utils/nnet-create.py \
        --set "$vars" \
        $encoder_conf \
        $outdir/init/encoder.mdl || exit_msg "Failed to create the VAE encoder"

    vars="latent_dim=$vae_hmm_latent_dim, \
    encoder_out_dim=$vae_hmm_encoder_out_dim, \
    dim_out=$feadim"
    python utils/nnet-create.py \
        --set "$vars" \
        $decoder_conf \
        $outdir/init/decoder.mdl || exit_msg "Failed to create the VAE decoder"

    vars="dim_in=$vae_hmm_latent_dim"
    python utils/nflow-create.py \
        --set "$vars" \
        $nflow_conf \
        $outdir/init/nflow.mdl || exit_msg "Failed to create the VAE norm. flow"

    python utils/create_emission.py \
        --dim $vae_hmm_latent_dim \
        $vae_hmm_emissions_conf \
        $outdir/init/emissions.mdl || exit_msg "Failed to create the emissions"

    python utils/vae-create.py \
        --encoder-cov-type $vae_hmm_encoder_cov_type \
        --decoder-cov-type $vae_hmm_decoder_cov_type \
        $train_datadir/feats.stats.npz \
        $vae_hmm_encoder_out_dim \
        $vae_hmm_latent_dim \
        $outdir/init/encoder.mdl \
        $outdir/init/nflow.mdl \
        $outdir/init/emissions.mdl \
        $outdir/init/decoder.mdl \
        $outdir/model_0.mdl || exit_msg "Failed to create the VAE"
else
    echo "Initial model already created: $outdir/model_0.mdl"
fi
exit 0

if [ ! -f $outdir/final.mdl ]; then
    echo "Training..."

    python utils/vae-hmm-train.py \
        --training_type $vae_hmm_training_type \
        --lrate $vae_hmm_lrate \
        --lrate-nnet $vae_hmm_lrate_nnet \
        --batch-size $vae_hmm_batch_size \
        --epochs $vae_hmm_epochs \
        --fast-eval \
        $train_datadir/feats.npz \
        $outdir/labels.int \
        $outdir/0.mdl \
        $train_datadir/feats.stats.npz \
        $outdir
else
    echo "The model is already trained. Final model is: $outdir/final.dml"
fi

