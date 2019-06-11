#!/bin/bash

if [ $# -ne 1 ]; then
    echo "$0 <out-dir>"
    exit 1;
fi

outdir=$1
mkdir -p $outdir

if [ -f $outdir/.done ]; then
    echo "data already prepared"
    exit 0
fi

# BUT
rootdir=/mnt/matylda2/data/GLOBALPHONE

# Default languages
languages="PO GE SP FR"
echo "Extracting languages: $languages"

# Assume script is being called in the recipe directory.
dir=$(dirname $0)

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
    for x in dev eval train; do
        mv -f $dir/data/$lang/$x $outdir/$lang
        cat $outdir/$lang/$x/wav.scp | awk '{print $1}' \
            > $outdir/$lang/$x/uttids || exit 1
    done
    mv -f $outdir/$lang/eval $outdir/$lang/test
done

date > $outdir/.done

