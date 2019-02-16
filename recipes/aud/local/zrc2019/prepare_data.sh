#!/usr/bin/env bash


# Path to the data.
datapath=samples.tar.bz2


if [ $# -ne 1 ]; then
    echo "usage: $0 <out-datadir>"
    exit 1
fi

datadir=$1


# Download the data.
if [ ! -f $datadir/local/.done ]; then
    echo "Extracting data..."
    mkdir -p $datadir/local/
    tar -xvzf $datapath -C $datadir/local
    date > $datadir/local/.done
else
    echo "Data already extracted. Skipping."
fi

mkdir -p $datadir/train

function scp_line {
    wav=$0
    uttid=$(basename $wav)
    uttid=${uttid%.*}
    echo $uttid $wav
}
export -f scp_line


if [ ! -f $datadir/train/uttids ]; then
    echo "creating wavs.scp file..."
    find $datadir/local/samples -name '*wav' \
        -exec bash -c 'scp_line "$0"' {} {} \; \
        | sort | uniq > $datadir/train/wavs.scp

    echo "creating uttids file..."
    cat $datadir/train/wavs.scp | awk '{print $1}' >$datadir/train/uttids
else
    echo "training data already prepared. Skipping."
fi

exit 0


function scp_line {
    wav=$0
    uttid=$(basename $wav)
    uttid=${uttid%.*}
    echo $uttid $wav
}
export -f scp_line


# Create the uttids/wavs.scp files for each data set.
chmod +x $datadir/local/mboshi/script/fix_wav.sh
for x in train dev; do
    if [ ! -f $datadir/$x/uttids ]; then
        echo "Fixing WAV files for the $x data set..."
        mkdir -p $datadir/local/mboshi/full_corpus_newsplit/${x}.fixed
        find $datadir/local/mboshi/full_corpus_newsplit/$x -name '*wav' \
            -exec $datadir/local/mboshi/script/fix_wav.sh {} \
            $datadir/local/mboshi/full_corpus_newsplit/${x}.fixed \;


        echo "Creating wavs.scp/uttids files for the $x data set..."
        mkdir -p $datadir/$x
        find $datadir/local/mboshi/full_corpus_newsplit/${x}.fixed -name '*wav' \
            -exec bash -c 'scp_line "$0"' {} {} \; \
            | sort | uniq > $datadir/$x/wavs.scp
        cat $datadir/$x/wavs.scp | awk '{print $1}' >$datadir/$x/uttids
    else
        echo "Dataset \"${x}\" already prepared. Skipping."
    fi
done

