#!/bin/bash

if [ $# -ne 5 ];then
    echo "Prepare train and test data, and language directory"
    echo "$0: phonemes_set, train_dir test_dir langdir"
    echo "eg: $0 phones.txt data/train data/test data/lang"
    exit 1
fi


phones=$1
traindir=$2
testdir=$3
langdir=$4
nstate_per_phone=$5
rootdir=`pwd`

[ ! -d $langdir ] && mkdir -p $langdir && exit 1;

# Skipping feature extraction part


# Convert transcription from phonemes sequences to integers sequences
awk 'BEGIN{i=0}{print $1" "i;i++}' $phones > $langdir/phones.txt

for d in $traindir $testdir; do
    python3 steps/prepare_labels.py $langdir/phones.txt \
        $d/phones.text \
        $nstate_per_phone
    cd $d/tmp
    zip `pwd`/../phones.int.npz *.npy
    cd $rootdir
    rm -r $d/tmp
done
