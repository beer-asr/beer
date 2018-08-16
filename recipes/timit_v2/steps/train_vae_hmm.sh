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
decoder_conf=$vae_hmm_encoder_conf
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

encoder_conf=$vae_hmm_encoder_conf
decoder_conf=$vae_hmm_decoder_conf
emissions_conf=$vae_hmm_emissions_conf
nflow_conf=$vae_hmm_normalizing_flow_conf


# Check if all the inputs are here.
[ -f "$encoder_conf" ] || exit_msg "File not found: $encoder_conf"
[ -f "$decoder_conf" ] || exit_msg "File not found: $decoder_conf"
[ -f "$emissions_conf" ] || exit_msg "File not found: $emissions_conf"
[ -f "$nflow_conf" ] || exit_msg "File not found: $nflow_conf"


mkdir -p "$outdir"/init

# Create the components of the VAE.
python utils/nnet-create.py \
    --set dim_in=$feadim,dim_out=$vae_hmm_encoder_out_dim \
    $encoder_conf \
    $outdir/init/encoder.mdl || exit_msg "Failed to create the VAE encoder"

python utils/nnet-create.py \
    --set latent_dim=$vae_hmm_latent_dim,encoder_out_dim=$vae_hmm_encoder_out_dim,dim_out=$feadim \
    $decoder_conf \
    $outdir/init/decoder.mdl || exit_msg "Failed to create the VAE decoder"

python utils/nflow-create.py \
    --set dim_in=$vae_hmm_latent_dim \
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
    $outdir/vae_emissions.mdl || exit_msg "Failed to create the VAE"
exit 0


# If the features are already created, do nothing.
if [ ! -f "$datadir"/feats.npz ]; then
    # Split the scp file into chunks to parallelize the features
    # extraction.
    mkdir -p "$datadir"/split
    pushd "$datadir"/split > /dev/null
    cp ../wav.scp ./
    split --numeric-suffixes=1 -n l/$fea_njobs ./wav.scp
    popd > /dev/null
    trap 'rm -rf "$datadir/split"' EXIT

    # Cleanup the log files.
    rm -f "$logdir"/extract-features.out.*

    tmpdir=$(mktemp -d "$datadir"/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT

    # Extract the features on the SGE.
    cmd="python utils/extract-features.py $fea_conf $tmpdir"
    qsub \
        -N "beer-extract-features" \
        -cwd \
        -j y \
        -o "$logdir"/extract-features.out.'$TASK_ID' \
        -t 1-$fea_njobs \
        -sync y \
        $fea_sge_opts \
        utils/jobarray.qsub "$cmd" "$datadir"/split || exit 1

    # Create the "npz" archives.
    find "$tmpdir" -name '*npy' | zip -j -@ "$datadir"/feats.npz > /dev/null
else
    echo "Features already extracted in: $datadir/feats.npz."
fi


if [ ! "$datadir"/feats.stats.npz ]; then
    python utils/compute_data_stats.py \
        $datadir/feats.npz $datadir/feats.stats.npz
else
   echo "Features statistics already computed in: $datadir/feats.stats.npz"
fi
