#!/bin/bash

set -e

if [ $# -ne 2 ]; then
    echo "$0 <out-dir> <lang>"
    exit 1;
fi


outdir=$1
languages=$2
mkdir -p $outdir

if [ -f $outdir/.done ]; then
    echo "data already prepared"
    exit 0
fi


# Assume script is being called in the recipe directory.
dir=$(dirname $0)

# BUT
rootdir=/mnt/matylda2/data/GLOBALPHONE

## Kaldi data preparation.
# 1) Extract the audio
pushd $dir
./gp_data_prep.sh --config-dir=conf --corpus-dir=$rootdir \
    --languages="$languages" || exit 1

# 2) Extract the lexicon
./gp_dict_prep.sh $rootdir $languages || exit 1

popd

for lang in $languages; do
    for x in dev eval train; do
        echo "preparing phonetic transcription for $lang ($x)"
        python utils/prepare_phone_trans.py \
            --add-sil $dir/data/$lang/local/dict/lexicon.txt \
            $dir/data/$lang/$x/text \
            > $dir/data/$lang/$x/trans 2>/dev/null || exit 1
    done
done

for lang in $languages; do
    mkdir -p $outdir/$lang

    # Prepare the list of language specific units.
    mkdir -p $outdir/$lang/lang
    cat $dir/data/$lang/local/dict/silence_phones.txt | \
        awk '{print $1" non-speech-unit"}' \
        > $outdir/$lang/lang/units
    cat $dir/data/$lang/local/dict/nonsilence_phones.txt | \
        awk -v lang=$lang '{print lang"_"$1" speech-unit"}' \
        >> $outdir/$lang/lang/units

    # Mapping name -> language specific name.
    cat $dir/data/$lang/local/dict/silence_phones.txt | \
        awk '{print $1" "$1}' \
        > $dir/data/$lang/local/dict/mapping.txt
    cat $dir/data/$lang/local/dict/nonsilence_phones.txt | \
        awk -v lang=$lang '{print $1" "lang"_"$1}' \
        >> $dir/data/$lang/local/dict/mapping.txt

    for x in dev eval train; do
        mkdir -p $outdir/$lang/$x

        # Make language specific transcription.
        python utils/maptrans.py \
            $dir/data/$lang/local/dict/mapping.txt \
            $dir/data/$lang/$x/trans \
            > $outdir/$lang/$x/trans


        cat $dir/data/$lang/$x/trans | awk '{print $1}' \
            > $outdir/$lang/$x/uttids

        # Keep only the utterances for which there is a proper
        # transcription.
        cat $dir/data/$lang/$x/wav.scp | \
            grep -w -f $outdir/$lang/$x/uttids > $outdir/$lang/$x/wav.scp
    done

    # Rename "eval" to "test" for consistency with other corpora.
    if [ -d $outdir/$lang/eval ]; then
        mv -f $outdir/$lang/eval $outdir/$lang/test
    fi
done

date > $outdir/.done

