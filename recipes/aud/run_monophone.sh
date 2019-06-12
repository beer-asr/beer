#!/usr/bin/env bash

# Exit if one command fails.
set -e

########################################################################
## SETUP

## DIRECTORY STRUCTURE
datadir=data        # where will stored the corpus specific data (transcription, dictionary, ...)
feadir=features     # where will be stored the features
expdir=exp          # experiment directory where will be stored the models and the results

## DATA
db=globalphone      # name of the corpus (timit, mboshi, globalphone)
subset=FR           # subset of the corpus (mostly used for globalphone: FR, GE, ...)
train=train         # name of the train set (usually "train")
test=test           # name of the test set (usuall "test")

## FEATURES
feaname=mfcc

## MONOPHONE MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
ngauss=4            # number of Gaussian per state.
latent_dim=100      # latent dimension of the subspace model
epochs=40           # number of training epochs

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_61_to_39.txt"

########################################################################


# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db $subset


for x in $train $test; do
    echo "--> Extracting features for the $db/$x database"
    steps/extract_features.sh \
        conf/${feaname}.yml \
        $datadir/$db/$subset/$x \
        $feadir/$db/$subset/$x

    # Create a "dataset". This "dataset" is just an object
    # associating the features with their utterance id and some
    # other meta-data (e.g. global mean, variance, ...).
    echo "--> Creating dataset(s) for $db database"
    steps/create_dataset.sh $datadir/$db/$subset/$x \
        $feadir/$db/$subset/$x/${feaname}.npz \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl
done


# Monophone system training.
steps/monophone.sh \
    --prior $prior \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm_${ngauss}g.yml \
    data/$db/$subset/lang \
    data/$db/$subset/$train \
    $expdir/$db/$subset/datasets/$feaname/${train}.pkl \
    $epochs $expdir/$db/$subset/monophone_${feaname}_${ngauss}g_${prior}

exit 0

# Subspace HMM monophone training.
steps/subspace_monophone.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    --latent-dim $latent_dim \
    conf/hmm.yml \
    $expdir/$db/monophone_${feaname} \
    data/$db/$dataset \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/subspace_monophone_${feaname}_ldim${latent_dim}

olddb=$db
db=timit
echo "--> Training the subspace Bayesian AUD system"
steps/subspace_aud.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    --latent-dim $latent_dim \
    conf/hmm.yml \
    $expdir/$olddb/subspace_monophone_${feaname}_ldim${latent_dim}/gsm_30.mdl \
    $expdir/$db/aud \
    data/$db/$dataset \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/subspace_aud_${feaname}_ldim${latent_dim}_pdata${olddb}



exit 0

# Subspace HMM monophone training.
steps/subspace_monophone.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    --latent-dim $latent_dim \
    --classes data/$db/lang/classes \
    conf/hmm.yml \
    $expdir/$db/monophone_${feaname} \
    data/$db/$dataset \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/dsubspace_monophone_${feaname}_ldim${latent_dim}

