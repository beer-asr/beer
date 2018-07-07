#!/bin/bash

# Set the environment.
. path.sh

# VAE-PLDA experiment:
#   compare PLDA model with its non-linear counter-part the VAE-PLDA.
#

#######################################################################
## Setup.

# Paths.
dbname= # <- Set the name of the database contained in the "data" directory.
dbpath=data/${dbname}
train_archives=${dbpath}/train_archives
test_archives=${dbpath}/test_archives
outdir=exp/${dbname}

# Training (PLDA/VAE-PLDA).
lograte=100      # Logging rate
epochs=10        # number of epochs for the training.
bsize=5000       # Number of feature frame per batch.
lrate=.1         # Learning rate.
gpu="--use-gpu"  # Comment if you don't need the GPU for training.
pt_epochs=5      # number of epochs for the pre-training (VAE-PLDA only).
lrate_nnet=1e-3  # Learning rate for the encoder/decoder (VAE-PLDA only).
nsamples=5       # Number of samples for the expectation (VAE-PLDA only).

# Testing options.
gpu_accuracy="--use-gpu"
                     # Comment if you don't need the GPU for computing
                     # the accuracy.
bsize_accuracy=1000  # Batch-size to compute the accuracy. Note that
                     # this value will not change the accuracy but only
                     # the memory needed to compute it.
nsamples_accuracy=5  # Number of samples to compute the accuracy
                     # (VAE-PLDA only).


#######################################################################
## Utility functions.


function compute_stats {
    local archives=$1
    local outdir=$2
    if [ ! -f "${outdir}/dbstats.npz" ]; then
        echo "Computing stats..."
        python utils/compute-stats.py \
            ${archives} \
            ${outdir}/dbstats.npz || exit 1
    else
        echo "Statistics already computed. Skipping."
    fi
}


function create_model {
    conf=$1
    dbstats=$2
    model_output=$3
    if [ ! -f "${model_output}" ]; then
        echo "Creating the model..."
        python utils/create-model.py \
            "${conf}" \
            "${dbstats}" \
            "${model_output}" || exit 1
    else
        echo "Model already create. Skipping."
    fi
}


function train_plda {
    local options=$1
    local dbstats=$2
    local train_archives=$3
    local input_model=$4
    local outdir=$5

    if [ ! -f "${outdir}/final.mdl" ]; then
        echo "Training..."
        # Command to submit to the SGE.
        cmd="python utils/train-std-discrete-latent-model.py \
           ${options} \
           ${dbstats} \
           ${train_archives} \
           ${input_model} \
           ${outdir}/final.mdl"

        # Clear the log file.
        rm -f ${outdir}/training.log

        # Submit the command to the SGE.
        qsub \
            -l gpu=1,mem_free=4G,ram_free=4G,hostname='c*' \
            -cwd \
            -j y \
            -sync y \
            -o ${outdir}/training.log \
            utils/job.qsub \
            "${cmd}" || exit 1
    else
        echo "Model already trained. Skipping."
    fi
}

function pretrain_vaeplda {
    local options=$1
    local dbstats=$2
    local train_archives=$3
    local input_model=$4
    local outdir=$5

    if [ ! -f "${outdir}/final.mdl" ]; then
        echo "Pre-training..."
        # Command to submit to the SGE.
        cmd="python utils/train-vae-discrete-latent-model.py \
           ${options} \
           ${dbstats} \
           ${train_archives} \
           ${input_model} \
           ${outdir}/pretrained.mdl"

        # Clear the log file.
        rm -f ${outdir}/pretraining.log

        # Submit the command to the SGE.
        qsub \
            -l gpu=1,mem_free=4G,ram_free=4G,hostname='c*' \
            -cwd \
            -j y \
            -sync y \
            -o ${outdir}/pretraining.log \
            utils/job.qsub \
            "${cmd}" || exit 1
    else
        echo "Model already pre-trained. Skipping."
    fi
}


function train_vaeplda {
    local options=$1
    local dbstats=$2
    local train_archives=$3
    local input_model=$4
    local outdir=$5

    if [ ! -f "${outdir}/final.mdl" ]; then
        echo "Training..."
        # Command to submit to the SGE.
        cmd="python utils/train-vae-discrete-latent-model.py \
           ${options} \
           ${dbstats} \
           ${train_archives} \
           ${input_model} \
           ${outdir}/final.mdl"

        # Clear the log file.
        rm -f ${outdir}/training.log

        # Submit the command to the SGE.
        qsub \
            -l gpu=1,mem_free=4G,ram_free=4G,hostname='c*' \
            -cwd \
            -j y \
            -sync y \
            -o ${outdir}/training.log \
            utils/job.qsub \
            "${cmd}" || exit 1
    else
        echo "Model already trained. Skipping."
    fi
}


function accuracy_plda {
    local options=$1
    local test_archives=$2
    local model=$3
    local outdir=$4

    if [ ! -f "${outdir}/accuracy" ]; then
        echo "Computing accuracy..."
        # Command to submit to the SGE.
        cmd="python utils/accuracy-std-discrete-latent-model.py \
           ${options} \
           ${test_archives} \
           ${model} \
           ${outdir}/accuracy"

        # Clear the log file.
        rm -f ${outdir}/accuracy.log

        # Submit the command to the SGE.
        qsub \
            -l gpu=1,mem_free=4G,ram_free=4G,hostname='c*' \
            -cwd \
            -j y \
            -sync y \
            -o ${outdir}/accuracy.log \
            utils/job.qsub \
            "${cmd}" || exit 1
    else
        echo "Accuracy already computed. Skipping."
    fi
    cat "${outdir}/accuracy"
}


function accuracy_vaeplda {
    local options=$1
    local test_archives=$2
    local model=$3
    local outdir=$4

    if [ ! -f "${outdir}/accuracy" ]; then
        echo "Computing accuracy..."
        # Command to submit to the SGE.
        cmd="python utils/accuracy-vae-discrete-latent-model.py \
           ${options} \
           ${test_archives} \
           ${model} \
           ${outdir}/accuracy"

        # Clear the log file.
        rm -f ${outdir}/accuracy.log

        # Submit the command to the SGE.
        qsub \
            -l gpu=1,mem_free=4G,ram_free=4G,hostname='c*' \
            -cwd \
            -j y \
            -sync y \
            -o ${outdir}/accuracy.log \
            utils/job.qsub \
            "${cmd}" || exit 1
    else
        echo "Accuracy already computed. Skipping."
    fi
    cat "${outdir}/accuracy"
}


#######################################################################
## Recipe.

# Step counter for the output logging message.
step=0

# Make sure the output directory of the experiment exists.
mkdir -p ${outdir}

##############################
# Statistics of the database #
##############################
echo "($((++step))) Computing statistics of the database..."
compute_stats \
    "${train_archives}" \
    "${outdir}"  || exit 1

##############
# PLDA Model #
##############
echo "($((++step))) PLDA model"

modelname=plda
model_outdir="${outdir}/${modelname}/epochs${epochs}_bsize${bsize}_lrate${lrate}"
options="\
--epochs ${epochs}  \
--batch-size ${bsize}  \
${gpu}  \
--lrate ${lrate}  \
--logging-rate ${lograte}  \
--dir-tmp-models ${model_outdir}/tmp_models \
"

accuracy_options="\
${gpu_accuracy} \
--batch-size ${bsize_accuracy} \
"

mkdir -p "${model_outdir}"
mkdir -p "${model_outdir}/tmp_models"

create_model \
    "data/${dbname}/${modelname}.yml" \
    "${outdir}/dbstats.npz" \
    "${outdir}/${modelname}/init.mdl" || exit 1

train_plda \
    "${options}" \
    "${outdir}/dbstats.npz" \
    "data/${dbname}/train_archives" \
    "${outdir}/${modelname}/init.mdl" \
    "${model_outdir}" || exit 1

accuracy_plda \
    "${accuracy_options}" \
    data/${dbname}/test_archives \
    "${model_outdir}/final.mdl" \
    "${model_outdir}" || exit 1


##################
# VAE-PLDA Model #
##################

echo "($((++step))) VAE-PLDA model"

modelname=vaeplda
model_outdir="${outdir}/${modelname}/ptepochs${pt_epochs}_epochs${epochs}_bsize${bsize}_lrate${lrate}_nsamples${nsamples}"
pretraining_options="\
--epochs ${pt_epochs}  \
--batch-size ${bsize}  \
${gpu}  \
--lrate ${lrate}  \
--lrate-nnet ${lrate_nnet} \
--logging-rate ${lograte}  \
--dir-tmp-models ${model_outdir}/tmp_models_pretraining \
--nsamples ${nsamples} \
--kl-weight 0. \
"

training_options="\
--epochs ${epochs}  \
--batch-size ${bsize}  \
${gpu}  \
--lrate ${lrate}  \
--lrate-nnet ${lrate_nnet} \
--logging-rate ${lograte}  \
--dir-tmp-models ${model_outdir}/tmp_models_training \
--nsamples ${nsamples} \
--kl-weight 1. \
"

accuracy_options="\
${gpu_accuracy} \
--batch-size ${bsize_accuracy} \
--nsamples ${nsamples_accuracy} \
"


mkdir -p "${model_outdir}"
mkdir -p "${model_outdir}/tmp_models_pretraining"
mkdir -p "${model_outdir}/tmp_models_training"

echo "outdir: $outdir"
create_model \
    "data/${dbname}/${modelname}.yml" \
    "${outdir}/dbstats.npz" \
    "${outdir}/${modelname}/init.mdl" || exit 1

pretrain_vaeplda \
    "${pretraining_options}" \
    "${outdir}/dbstats.npz" \
    "data/${dbname}/train_archives" \
    "${outdir}/${modelname}/init.mdl" \
    "${model_outdir}" || exit 1

train_vaeplda \
    "${training_options}" \
    "${outdir}/dbstats.npz" \
    "data/${dbname}/train_archives" \
    "${model_outdir}/pretrained.mdl" \
    "${model_outdir}" || exit 1

accuracy_vaeplda \
    "${accuracy_options}" \
    data/${dbname}/test_archives \
    "${model_outdir}/final.mdl" \
    "${model_outdir}" || exit 1

