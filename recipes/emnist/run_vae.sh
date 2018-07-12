#!/bin/sh

# Set the environment.
source "$(pwd)/path.sh"

# SGE options.
# JHU/CLSP cluster.
#sge_opts="-l gpu=1,mem_free=1G,ram_free=1G,hostname='c*'"
# BRNO/FIT cluster.
sge_opts="-l gpu=1,mem_free=1G,ram_free=1G"

# Database to use ("digits" or "letters").
dbname=digits

# Model configuration file.
modelname=vae
modelconf="conf/${modelname}.yml"

# Output directory of the experiment.
outdir="exp/${dbname}/${modelname}"

# Prepare the data.
local/prepare_emnist_data.sh || exit 1

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
steps/train-vae-discrete-latent-model.sh \
    --use-gpu \
    --lograte=10 \
    --pt-epochs=10 \
    --epochs=100 \
    --lrate=.1 \
    --lrate-nnet=1e-3 \
    --nsamples=5 \
    --unsupervised \
    -- \
    "${sge_opts}" \
    "${outdir}/init.mdl" \
    "${outdir}/dbstats.npz" \
    "data/${dbname}/train/archives" \
    "${outdir}" || exit  1

