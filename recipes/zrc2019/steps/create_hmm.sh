#!/usr/bin/env bash

args=()
outdir="./"
phone_map=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --outdir)
            outdir="$2"
            shift
            shift
            ;;
        --phone-map)
            phone_map="$2"
            shift
            shift
            ;;
        *) # unknown option
            args+=("$1")
            shift
        ;;
    esac
done
set -- "${args[@]}"

if [ $# -ne 3 ]; then
    echo "usage: [--outdir DIR] [--phone-map MAP] <model-conf> <dataset> <out-model>"
    exit 1
fi

modelconf=$1
dataset=$2
outmodel=$3
mkdir -p $outdir


# Create the units' HMM.
if [ ! -f $outmodel ]; then
    # Create the phones' HMM.
    beer hmm mkphones -d $dataset $modelconf $outdir/hmms.mdl || exit 1

    # Make the phone-loop graph.
    beer hmm phonelist $outdir/hmms.mdl | \
        beer hmm mkphoneloopgraph - \
        $outdir/ploop_graph.pkl || exit 1

    # Add the HMMs to the graph and compile it.
    beer hmm mkdecodegraph $outdir/ploop_graph.pkl $outdir/hmms.mdl \
        $outdir/decode_graph.pkl || exit 1

    # Create the final HMM.
    beer hmm mkphoneloop $outdir/decode_graph.pkl $outdir/hmms.mdl \
        $outmodel
else
    echo "HMM already created. Skipping."
fi

