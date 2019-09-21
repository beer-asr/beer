#!/usr/bin/env bash


# git URL to the Mboshi data repository.
url="https://github.com/besacier/mboshi-french-parallel-corpus.git"


if [ $# -ne 1 ]; then
    echo "usage: $0 <out-datadir>"
    exit 1
fi

datadir=$1
cwd=$(dirname $0)

# Download the data.
if [ ! -d $datadir/local/mboshi ]; then
    echo "Downloading the MBOSHI database..."
    git clone $url $datadir/local/mboshi
else
    echo "Data already downloaded. Skipping."
fi


function scp_line {
    wav=$0
    uttid=$(basename $wav)
    uttid=${uttid%.*}
    echo $uttid $wav
}
export -f scp_line


# Create the uttids/wav.scp files for each data set.
chmod +x $datadir/local/mboshi/script/fix_wav.sh
for x in train dev; do
    if [ ! -f $datadir/$x/uttids ]; then
        echo "Fixing WAV files for the $x data set..."
        mkdir -p $datadir/local/mboshi/full_corpus_newsplit/${x}.fixed
        find $datadir/local/mboshi/full_corpus_newsplit/$x -name '*wav' \
            -exec $datadir/local/mboshi/script/fix_wav.sh {} \
            $datadir/local/mboshi/full_corpus_newsplit/${x}.fixed \;

        echo "Creating wav.scp/uttids files for the $x data set..."
        mkdir -p $datadir/$x
        find $datadir/local/mboshi/full_corpus_newsplit/${x}.fixed -name '*wav' \
            -exec bash -c 'scp_line "$0"' {} {} \; \
            | sort | uniq > $datadir/$x/wav.scp
        cat $datadir/$x/wav.scp | awk '{print $1}' >$datadir/$x/uttids

        cat $cwd/mboshi.ali | grep -w -f $datadir/$x/uttids \
            > $datadir/$x/ali
    else
        echo "Dataset \"${x}\" already prepared. Skipping."
    fi
done
