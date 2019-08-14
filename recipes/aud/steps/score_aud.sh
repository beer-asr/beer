#!/usr/bin/env bash

. path.sh

nargs=3

while [[ $# -gt $nargs ]]; do
    case $1 in
        --mapping)
        mapping="--mapping $2"
        shift
        shift
        ;;
        --au-mapping)
        au_mapping="$2"
        shift
        shift
        ;;
        *)
        echo "unknown option: $1"
        exit 1
    esac
done

if [ $# -ne $nargs ]; then
    echo "usage: $0 [OPTS] <ref-ali> <hyp-ali> <out-dir>"
    echo ""
    echo "Score the data-driven acoustic unit transcription"
    echo ""
    exit 1
fi

ref_ali=$1
hyp_ali=$2
outdir=$3
mkdir -p $outdir


if [ ! -f $oudir/au_phone_counts.yml ]; then
    echo "creating a mapping acoustic unit -> phone"
    python utils/maxoverlap_mapping.py $ref_ali $hyp_ali \
        --counts $outdir/au_phone_counts.yml $outdir/au_phone
    au_mapping=$outdir/au_phone
fi

if [ ! -f $outdir/eq_per ]; then
   echo "mapping the acoustic unit to phones"
   python utils/maptrans.py --unk '<unk>' $au_mapping $hyp_ali \
           > $outdir/au_phone_trans

   echo "computing the equivalent Phone Error Rate"
   python utils/ter.py --no-repeat $mapping $ref_ali \
       $outdir/au_phone_trans > $outdir/eq_per || exit 1
fi
echo "PER: $(tail -n 1 $outdir/eq_per)"

if [ ! -f $outdir/nmi ]; then
    echo "computing the NMI"
    python utils/nmi.py $outdir/au_phone_counts.yml > $outdir/nmi || exit 1
fi
echo "NMI: $(tail -n 1 $outdir/nmi)"


if [ ! -f $outdir/entropy_rate ]; then
    echo "evaluating the entropy rate"
    python utils/entropy_rate.py $hyp_ali > $outdir/entropy_rate || exit 1
fi
echo "entropy rate, perplexity: $(tail -n 1 $outdir/entropy_rate)"


if [ ! -f $outdir/phone_boundaries ]; then
    echo "evaluating the segmentation"
    python utils/score_boundaries.py $ref_ali $hyp_ali \
        > $outdir/phone_boundaries || exit 1
fi
echo "recall, precision, fscore: $(tail -n 1 $outdir/phone_boundaries)"

