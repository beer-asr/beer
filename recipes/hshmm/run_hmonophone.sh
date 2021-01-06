#!/usr/bin/env bash

# Exit if one command fails.
set -e

echo "$0 $@"
########################################################################
## SETUP

# Use learning rate of 5e-3
hsubspace_opts="--gsm-std-lrate 5e-3"

## DIRECTORY STRUCTURE
datadir=data        # where will stored the corpus specific data (transcription, dictionary, ...)
feadir=/mnt/scratch04/tmp/xyusuf00/features     # where will be stored the features
expdir=exp_hsubspace          # experiment directory where will be stored the models and the results

## DATA
train=train         # name of the train set (usually "train")
test="test"     # name of the test set (usually "test")

## FEATURES
feaname=mfcc

## MONOPHONE MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
ngauss=4            # number of Gaussian per state.
unit_latent_dim=100      # unit latent dimension                                                                                          
lang_latent_dim=6      # language latent subspace model
epochs=100           # number of training epochs
epochs_hmm=20        # number of training epoch for forced alignment model
parallel_njobs=1000  # number of jobs for accumulating statistics per language

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_48_to_39.txt"

########################################################################

[ -f ./kutils/parse_options.sh ] && . ./kutils/parse_options.sh
#if [ $# -lt 1 ] || [ $# -gt 2 ]; then
if [ $# -ne 1 ]; then
    echo "usage: $0 data_config"
    echo ""
    echo "Train the hyper-subspace for H-SHMM based AUD"
    echo ""
    echo "Examples:"
    echo "  $ $0 data.conf"
    echo ""
    exit 1
fi

db_subs_file=$1  # file of tuples corpus subset1, subset2, ... tuples
# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir

conf=$1
all_langs=""
for line in `cat $db_subs_file`; do
    db=${line%:*}
    echo "--> Preparing data for the $db database"
    subs=${line#*:}
    subset=`echo ${subs} | tr , ' '`
    local/$db/prepare_data.sh $datadir/$db "$subset"
    all_langs="$all_langs $subset"
done

all_langs=`python3 -c "print('_'.join('$all_langs'.split()))"`
train_datadirs=""
train_datasets=""

for line in `cat $db_subs_file`; do
    db=${line%:*}
    subs=${line#*:}
    subset=`echo ${subs} | tr , ' '`
    for lang in $subset; do
	train_datadirs="$lang,$datadir/$db/${lang}_subs/$train $train_datadirs"
	[ ! -d $datadir/$db/${lang}_subs ] && mkdir -p $datadir/$db/${lang}_subs
	if [ ! -f $datadir/$db/${lang}_subs/.done ]; then
	    # This part ensures that the model is trained with 1500 utterances
	    # To use the full set without changing the rest of the code, change the next command to:
	    # ln -s $lang $datadir/$db/${lang}_subs/
	    utils/add_corpus.sh --max-utts 1500 $datadir/$db/$lang $datadir/$db/${lang}_subs/
	    touch $datadir/$db/${lang}_subs/.done
	fi
	
	for x in $train $test; do
	    echo "--> Extracting features for the $db/$lang/$x database"
	    steps/extract_features.sh \
		--parallel-njobs $parallel_njobs \
		conf/${feaname}.yml \
		$datadir/$db/${lang}_subs/$x/wav.scp \
		$feadir/$db/${lang}/$x
	    
	    # Create a "dataset". This "dataset" is just an object
	    # associating the features with their utterance id and some
	    # other meta-data (e.g. global mean, variance, ...).
	    echo "--> Creating dataset(s) for $db database"
	    steps/create_dataset.sh $datadir/$db/${lang}_subs/$x \
				    $feadir/$db/${lang}/$x/${feaname}.npz \
				    $expdir/$db/$lang/datasets/$feaname/${x}.pkl
	done
	train_datasets="$lang,$expdir/$db/$lang/datasets/$feaname/${train}.pkl $train_datasets"
    done
done

all_hmmdirs=""
# Monophone system training for forced alignment.
for line in `cat $db_subs_file`; do
    db=${line%:*}
    subs=${line#*:}
    subset=`echo ${subs} | tr , ' '`

    for lang in $subset; do
	echo "--> training monophone for lang: $lang"
	steps/monophone.sh \
	    --prior $prior \
	    --parallel-opts "-l mem_free=1G,ram_free=1G" \
	    --parallel-njobs $parallel_njobs \
	    conf/hmm_${ngauss}g.yml \
	    data/$db/${lang}_subs/lang \
	    data/$db/${lang}_subs/$train \
	    $expdir/$db/$lang/datasets/$feaname/${train}.pkl \
	    $epochs_hmm \
	    $expdir/$db/$lang/monophone_${feaname}_${ngauss}g_${prior} &
	all_hmmdirs="$lang,$expdir/$db/$lang/monophone_${feaname}_${ngauss}g_${prior} $all_hmmdirs"
    done
done
wait

# H-SHMM hyper-subspace training.
[ -z $odir ] && odir=$all_langs
mkdir -p $expdir/$db/$odir
[ ! -f $expdir/$db/$odir/langs ] && echo $all_langs > $expdir/$db/$odir/langs
steps/hsubspace_monophone.sh \
    $hsubspace_opts \
    --lang-latent-dim $lang_latent_dim \
    --unit-latent-dim $unit_latent_dim \
    --parallel-opts "-l mem_free=1G,ram_free=1G -q all.q@@stable" \
    --parallel-njobs $parallel_njobs \
    conf/hmm_${ngauss}g.yml \
    "$all_hmmdirs" \
    "$train_datadirs" \
    "$train_datasets" \
    $epochs \
    $expdir/$db/$odir/hsubspace_monophone_${feaname}_${ngauss}g_${prior}_ldim${lang_latent_dim}_udim${unit_latent_dim}
