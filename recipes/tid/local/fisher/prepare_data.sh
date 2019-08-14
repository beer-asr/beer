#!/usr/bin/env bash

set -e 

if [ $# -ne 1 ]; then
    echo "usage: $0 <out-datadir>"
    exit 1
fi

cwdir=$(dirname $0)
datadir=$1/fisher

mkdir -p $datadir/{train,test}
cp $cwdir/topics $datadir

if [ ! -f "$datadir/.done" ]; then
    for x in train test; do 
        cp $cwdir/fisher_40c_${x}.flist $datadir/$x/docids 
        cat $cwdir/doclabels | grep -f $cwdir/fisher_40c_${x}.flist \
            > $datadir/$x/doclabels
    done
    date > "$datadir/.done"
else
    echo "FISHER data already prepared."
fi

