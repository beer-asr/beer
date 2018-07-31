#!/bin/sh

# Set the environment.
source "$(pwd)/path.sh"

# SGE options.
#sge_opts="-l gpu=1,mem_free=1G,ram_free=1G,hostname='c*'"  # JHU/CLSP cluster
sge_opts="-l gpu=1,mem_free=1G,ram_free=1G"  # Brno/FIT cluster

# Prepare the data. This script will prepare both "digits" and
# "letters" data set.
local/prepare_emnist_data.sh || exit 1

# Database to use ("digits" or "letters").
dbname=digits

# Model configuration file.
modelname=vae_convnet_elu_bernoulli_ldim64_gmm_scov
modelconf="conf/${modelname}.yml"

# Output directory of the experiment.
outdir="exp/${dbname}/${modelname}"

# Create the output directory
mkdir -p "${outdir}"

# Compute the stats of the data base for the initialization/training
# of the model.
steps/compute-stats.sh \
    "data/${dbname}/train/archives" \
    "${outdir}/dbstats.npz" || exit 1

# Creating the model.
steps/create-model.sh \
    "${modelconf}" \
    "${outdir}/dbstats.npz" \
    "${outdir}/init.mdl" || exit 1

# Training the model.
steps/train-vae-model.sh \
    --use-gpu \
    --lograte=100 \
    --pt-epochs=20 \
    --pt-lrate=.1 \
    --pt-lrate-nnet=1e-3 \
    --epochs=30 \
    --lrate=.1 \
    --lrate-nnet=1e-3 \
    -- \
    "${sge_opts}" \
    "${outdir}/init.mdl" \
    "${outdir}/dbstats.npz" \
    "data/${dbname}/train/archives" \
    "${outdir}" || exit  1

# Compute the accuracy of the model.
steps/accuracy-vae-discrete-latent-model.sh \
    --use-gpu \
    -- \
    "-l gpu=1,mem_free=1G,ram_free=1G" \
    "${outdir}/final.mdl" \
    "data/${dbname}/test/archives" \
    "${outdir}/results"

