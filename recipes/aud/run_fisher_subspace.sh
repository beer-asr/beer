#!/usr/bin/env bash

# Exit if one command fails.
set -e

########################################################################
## SETUP

## DIRECTORY STRUCTURE
datadir=data
feadir=/mnt/scratch04/tmp/iondel/features
expdir=exp_dp

## DATA
db=fisher
train=train
test=test

## FEATURES
feaname=mbn

## AUD MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
bigram_prior=hierarchical_dirichlet_process

ngauss=4        # number of Gaussian per state.
nunits=100      # maximum number of discovered units
epochs=20       # number of training epochs
gsm_init=exp_new/globalphone/FR_PO_GE_SP/subspace_monophone_mbn_4g_gamma_dirichlet_process_ldim100

# Extract the latent dimension (assuming something like "...ldim10_..."
tmp=${gsm_init##*ldim}
ldim=${tmp%%_*}

########################################################################


# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


#echo "--> Preparing data for the $db database"
#local/$db/prepare_data.sh $datadir/$db


echo "--> Preparing pseudo-phones \"language\" information"
mkdir -p data/$db/lang_aud

# The option "--non-speech-unit" will force the decoder to start and end
# each utterance by a specific acoustic unit named "sil". The
# unit can also be freely decoded within the utterance. If your data
# is not well segmented and you are not sure that most of your data
# start and end with non-speech sound it's safer to remove this option.
python utils/prepare_lang_aud.py \
    --non-speech-unit \
    $nunits > data/$db/lang_aud/units


for x in $train $test; do
    echo "--> Extracting features for the $db/$x database"
    steps/extract_features.sh \
        --parallel-opts "-l scratch4=1" \
        --parallel-njobs 10 \
        conf/${feaname}.yml $datadir/$db/$x $feadir/$db/$x

    # Create a "dataset". This "dataset" is just an object
    # associating the features with their utterance id and some
    # other meta-data (e.g. global mean, variance, ...).
    echo "--> Creating dataset(s) for $db database"
    steps/create_dataset.sh $datadir/$db/$x \
        $feadir/$db/$x/${feaname}.npz \
        $expdir/$db/datasets/$feaname/${x}.pkl
done

# AUD system training. You need to have a Sun Grid Engine like cluster
# (i.e. qsub command) to run it. If you have a different
# enviroment please see utils/parallel/sge/* to see how to adapt
# this recipe to you system.
echo "--> Train the unigram subspace Bayesian AUD system"
steps/subspace_aud.sh \
    --prior $prior \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 50 \
    conf/hmm_${ngauss}g.yml \
    $gsm_init \
    data/$db/lang_aud \
    data/$db/$train \
    $expdir/$db/datasets/$feaname/${train}.pkl \
    $epochs $expdir/$db/$subset/aud_subspace_${feaname}_${ngauss}g_${prior}_ldim${ldim}

exit 0

echo "--> Train the bigram AUD system"
steps/aud_bigram.sh \
    --prior $bigram_prior \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    $expdir/$db/$subset/aud_${feaname}_${ngauss}g_${prior}/final.mdl \
    data/$db/$train \
    $expdir/$db/datasets/$feaname/${train}.pkl \
    $epochs $expdir/$db/$subset/aud_bigram_${feaname}_${ngauss}g_${bigram_prior}

