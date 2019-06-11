#!/usr/bin/env bash

. path.sh

au_mapping=""
mapping=""
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
    echo "Options:"
    echo "  --au-mapping        generate a au->phone mapping"
    echo "  --mapping           phone mapping"
    echo ""
    exit 1
fi

ref_ali=$1
hyp_ali=$2
outdir=$3
mkdir -p $outdir


if [ -z "$au_mapping" ]; then
    echo "creating a mapping acoustic unit -> phone"
    python utils/maxoverlap_mapping.py $ref_ali $hyp_ali \
        --counts $outdir/au_phone_counts.yml $outdir/au_phone
    au_mapping=$outdir/au_phone
fi


if [ ! -f $outdir/.done_per ]; then
    echo "mapping the acoustic unit to phones"
    python utils/maptrans.py --unk '<unk>' $au_mapping $hyp_ali \
        > $outdir/au_phone_trans

    echo "computing the equivalent Phone Error Rate"
    python utils/ter.py --no-repeat $mapping $ref_ali \
        $outdir/au_phone_trans > $outdir/eq_per || exit 1

    touch $outdir/.done_per
fi
echo "eq. PER: $(tail -n 1 $outdir/eq_per)"

if [ ! -f $outdir/.done_enr ]; then
    echo "evaluating the entropy rate"
    python utils/entropy_rate.py $hyp_ali > $outdir/entropy_rate || exit 1
    touch $outdir/.done_enr
fi
echo "enrtropy rate, perplexity: $(tail -n 1 $outdir/entropy_rate)"

if [ ! -f $outdir/.done_pb ]; then
    echo "evaluating the segmentation"
    python utils/score_boundaries.py $mapping $ref_ali $outdir/au_phone_trans \
        > $outdir/phone_boundaries || exit 1

    touch $outdir/.done_pb
fi
echo "recall, precision, fscore: $(tail -n 1 $outdir/phone_boundaries)"

