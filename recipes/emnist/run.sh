#!/bin/sh

# Set the environment.
source "$(pwd)/path.sh"

# Database to use ("digits" or "letters").
dbname=digits

# Model configuration file.
modelname=vaeplda
modelconf="conf/${modelname}.yml"

# Output directory of the experiment.
outdir="exp/${dbname}/${modelname}"

# Prepare the data.
local/prepare_emnist_data.sh

# Create the output directory
mkdir -p "${outdir}"

# Compute the stats of the data base for the initialization/training
# of the model.
steps/compute-stats.sh \
    "data/${dbname}/train/archives" \
    "${outdir}/dbstats.npz"

# Creating the model.
steps/create-model.sh \
    "${modelconf}" \
    "${outdir}/dbstats.npz" \
    "${outdir}/init.mdl"

# Training the model.
steps/train-vae-discrete-latent-model.sh \
    --use-gpu \
    --lograte=10 \
    --pt-epochs=5 \
    --epochs=50 \
    --lrate=.1 \
    --lrate-nnet=1e-3 \
    --nsamples=5 \
    -- \
    "-l gpu=1,mem_free=1G,ram_free=1G" \
    "${outdir}/init.mdl" \
    "${outdir}/dbstats.npz" \
    "data/${dbname}/train/archives" \
    "${outdir}"


# Compute the accuracy of the model.
steps/accuracy-vae-discrete-latent-model.sh \
    --use-gpu \
    --nsamples=5 \
    -- \
    "-l gpu=1,mem_free=1G,ram_free=1G" \
    "${outdir}/final.mdl" \
    "data/${dbname}/test/archives" \
    "${outdir}/results"

