#!/usr/bin/env bash

if [ $# -ne 3 ]; then
    echo "usage: <fea-conf> <data-dir> <out-fea-dir>"
    exit 1
fi

conf=$1
datadir=$2
feadir=$3
feaname=$(basename $conf)
feaname="${feaname%.*}"

mkdir -p $feadir


if [ ! -f $feadir/${feaname}.npz ]; then
    echo "extracting features in $feadir/${feaname}.npz"

    mkdir -p $feadir/${feaname}_tmp

    # Extract the features.
    beer features extract $conf $datadir/wavs.scp \
        $feadir/${feaname}_tmp || exit 1

    # Put all the features files into a single archive.
    beer features archive $feadir/${feaname}_tmp $feadir/${feaname}.npz

    # We don't need the original features anymore as they are stored in
    # the archive.
    rm -fr $feadir/${feaname}_tmp
else
    echo "Features already extracted ($feadir/${feaname}.npz). Skipping."
fi

