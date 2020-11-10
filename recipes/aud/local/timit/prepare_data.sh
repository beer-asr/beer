#!/bin/bash

# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
#           2018   Brno University of Technology (Author: Lucas Ondel)
# Apache 2.0.
#
# This is a slightly adapted version of the original Kaldi script.
#

set -e

# JHU
#rootdir=/export/corpora5/LDC/LDC93S1/timit/TIMIT

# BUT
rootdir=/mnt/matylda2/data/TIMIT/timit


if [ $# -ne 1 ]; then
    echo "$0 <out-dir>"
    exit 1;
fi

conf=$(dirname $0)
outdir=$1
dir=$(pwd)/$outdir/local
mkdir -p $outdir/local

[ -f $outdir/.done ] && echo "Data already prepared. Skipping." && exit 0;

[ -f $conf/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";
[ -f $conf/phones.60-48-39.map ] || error_exit "$PROG: phones.60-48-39.map not found."


# First check if the train & test directories exist (these can either
# be upper- or lower-cased
if [ ! -d $rootdir/TRAIN -o ! -d $rootdir/TEST ] \
    && [ ! -d $rootdir/train -o ! -d $rootdir/test ]; then
    echo "$0: Spot check of command line argument failed"
    echo "Command line argument must be absolute pathname to TIMIT directory"
    echo "with name like /export/corpora5/LDC/LDC93S1/timit/TIMIT"
    exit 1
fi

# Now check what case the directory structure is
uppercased=false
train_dir=train
test_dir=test
if [ -d $rootdir/TRAIN ]; then
    uppercased=true
    train_dir=TRAIN
    test_dir=TEST
fi

tmpdir=$(mktemp -d /tmp/beer.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 24-speaker core
# test set and the 50-speaker development set must be supplied to the
# script. All speakers in the 'train' directory are used for training.
if $uppercased; then
    tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
    tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
    ls -d "$rootdir"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
else
    tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
    tr '[:upper:]' '[:lower:]' < $conf/test_spk.list > $tmpdir/test_spk
    ls -d "$rootdir"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
fi

for x in train dev test; do
    # First, find the list of audio files (use only si & sx
    # utterances). Note: train & test sets are under different
    # directories, but doing find on both and grepping for the speakers
    # will work correctly.
    find $rootdir/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WAV' \
         | grep -f $tmpdir/${x}_spk > $dir/${x}_sph.flist

         sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' $dir/${x}_sph.flist \
             > $tmpdir/${x}_sph.uttids
         paste $tmpdir/${x}_sph.uttids $dir/${x}_sph.flist \
             | sort -k1,1 > $dir/${x}_sph.scp

         cat $dir/${x}_sph.scp | awk '{print $1}' > $dir/${x}.uttids

    # Now, Convert the transcripts into our format (no normalization
    # yet) Get the transcripts: each line of the output contains an
    # utterance ID followed by the transcript.
     find $rootdir/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.PHN' \
         | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist
     sed -e 's:.*/\(.*\)/\(.*\).PHN$:\1_\2:i' $tmpdir/${x}_phn.flist \
         > $tmpdir/${x}_phn.uttids
     while read line; do
         [ -f $line ] || error_exit "Cannot find transcription file '$line'";
         cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
     done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
     paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
         | sort -k1,1 > $dir/${x}.trans

    # Do normalization steps.
    cat $dir/${x}.trans | python $conf/timit-norm-trans.py \
        --map-60-48  $conf/phones.60-48-39.map \
        | sort > $dir/${x}.text || exit 1;

    # Create wav.scp
    awk '{printf("%s sph2pipe -f wav %s |\n", $1, $2);}' < $dir/${x}_sph.scp \
        > $dir/${x}_wav.scp

    # Make the utt2spk and spk2utt files.
    cut -f1 -d'_'  $dir/${x}.uttids | paste -d' ' $dir/$x.uttids - > $dir/$x.utt2spk

    # Extract the human alignments.
    python $conf/phone_ali.py $tmpdir/${x}_phn.flist \
        > $dir/${x}.human_alignments || exit

    mkdir -p $outdir/${x}
    cp $dir/${x}.human_alignments $outdir/$x/ali
    cp $dir/$x.uttids $outdir/$x/uttids
    cp $dir/${x}_wav.scp $outdir/$x/wav.scp
    cp $dir/${x}.text $outdir/$x/trans
done

mkdir -p $outdir/full
for x in train dev test; do
    cat $outdir/${x}/ali >> $outdir/full/ali
    cat $outdir/${x}/uttids >> $outdir/full/uttids
    cat $outdir/${x}/wav.scp >> $outdir/full/wav.scp
    cat $outdir/${x}/trans >> $outdir/full/trans
done


mkdir -p $outdir/lang
python $conf/timit_lang_prep.py $outdir/lang "$conf/phones.60-48-39.map"
cat local/timit/phones.60-48-39.map | awk '{print $1" "$3}' \
    > $outdir/lang/phones_61_to_39.txt || exit 1


date > $outdir/.done
echo "Data preparation succeeded"
