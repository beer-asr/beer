#!/usr/bin/env bash

# Exit if one command fails.
set -e

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <corpus> [<subset>]"
    echo ""
    echo "Build a monophone HMM based phone-recognizer"
    echo ""
    echo "Examples:"
    echo "  $ $0 timit"
    echo "  $ $0 globalphone FR"
    echo ""
    exit 1
fi

########################################################################
## SETUP

## DIRECTORY STRUCTURE
datadir=data        # where will stored the corpus specific data (transcription, dictionary, ...)
feadir=/mnt/scratch04/tmp/xyusuf00/features     # where will be stored the features
expdir=exp_spl          # experiment directory where will be stored the models and the results

## DATA
db=$1               # name of the corpus (timit, mboshi, globalphone)
subset=$2           # subset of the corpus (mostly used for globalphone: FR, GE, ...)
train=train         # name of the train set (usually "train")
test="dev test"     # name of the test set (usuall "test")

## FEATURES
feaname=mfcc

## MONOPHONE MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
ngauss=4            # number of Gaussian per state.
unit_latent_dim=2      # latent dimension of the subspace model                                                                                          
lang_latent_dim=2      # latent dimension of the subspace model
epochs=50           # number of training epochs

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_48_to_39.txt"

########################################################################


# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db "$subset"

all_langs=`python3 -c "print('_'.join('$subset'.split()))"`
# [ ! -d $datadir/$db/$all_langs ] && mkdir -p $datadir/$db/$all_langs
# if [ ! -f $datadir/$db/$all_langs/.done ]; then
#    echo "--> Combining data sets"
#    for lng in $subset; do
#        utils/add_corpus.sh --max-utts 1500 $datadir/$db/$lng $datadir/$db/$all_langs
#    done
#    touch $datadir/$db/$all_langs/.done
# fi

# if [ ! -f $datadir/$db/$all_langs/lang/unit_to_lang ]; then
#     echo "--> Creating unit to language map"
#     awk -F ' ' '!/non/ {print $1}' $datadir/$db/$all_langs/lang/units | \
# 	awk -F _ '{print $0, $1}' > $datadir/$db/$all_langs/lang/unit_to_lang
# fi

train_datadirs=""
train_datasets=""
for lang in $subset; do
    train_datadirs="$lang,$datadir/$db/${lang}_subs/$train $train_datadirs"
    [ ! -d $datadir/$db/${lang}_subs ] && mkdir -p $datadir/$db/${lang}_subs
    if [ ! -f $datadir/$db/${lang}_subs/.done ]; then
	utils/add_corpus.sh --max-utts 1500 $datadir/$db/$lang $datadir/$db/${lang}_subs/
	touch $datadir/$db/${lang}_subs/.done
    fi

    for x in $train $test; do
	echo "--> Extracting features for the $db/$lang/$x database"
	steps/extract_features.sh \
	    --parallel-njobs 100 \
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

all_hmmdirs=""
# Monophone system training.
for lang in $subset; do
    echo "--> training monophone for lang: $lang"
    steps/monophone.sh \
	--prior $prior \
	--parallel-opts "-l mem_free=1G,ram_free=1G" \
	--parallel-njobs 100 \
	conf/hmm_${ngauss}g.yml \
	data/$db/${lang}_subs/lang \
	data/$db/${lang}_subs/$train \
	$expdir/$db/$lang/datasets/$feaname/${train}.pkl \
	$epochs \
	$expdir/$db/$lang/monophone_${feaname}_${ngauss}g_${prior}
    all_hmmdirs="$lang,$expdir/$db/$lang/monophone_${feaname}_${ngauss}g_${prior} $all_hmmdirs"
done

# Subspace HMM monophone training.
steps/hsubspace_monophone.sh \
    --lang-latent-dim $lang_latent_dim \
    --unit-latent-dim $unit_latent_dim \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 100 \
    conf/hmm_${ngauss}g.yml \
    "$all_hmmdirs" \
    "$train_datadirs" \
    "$train_datasets" \
    $epochs \
    $expdir/$db/$all_langs/subspace_monophone_${feaname}_${ngauss}g_${prior}_ldim${lang_latent_dim}_udim${unit_latent_dim}


exit 0

for x in $test; do
    outdir=$expdir/$db/$subset/subspace_monophone_${feaname}_${ngauss}g_${prior}/decode_perframe/$x

    echo "--> Decoding $db/$x dataset"
    steps/decode.sh \
        --per-frame \
        --parallel-opts "-l mem_free=1G,ram_free=1G" \
        --parallel-njobs 30 \
        $expdir/$db/$subset/subspace_monophone_${feaname}_${ngauss}g_${prior}/final.mdl \
        data/$db/$subset/$x \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl \
        $outdir

    echo "--> Scoring $db/$x dataset"
    steps/score_aud.sh \
        $au_mapping \
        $mapping \
        data/$db/$subset/$x/ali \
        $outdir/trans \
        $outdir/score
done

