#!/usr/bin/env bash

. path.sh

mapping=""
nargs=3

while [[ $# -gt $nargs ]]; do
    case $1 in
      --mapping)
      mapping="--mapping $2"
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
    echo "Score the token error rate of a transcription"
    echo ""
    echo "Options:"
    echo "  --mapping           phone mapping"
    echo ""
    exit 1
fi

ref_ali=$1
hyp_ali=$2
outdir=$3
mkdir -p $outdir


if [ ! -f $outdir/.done_per ]; then
    echo "mapping the acoustic unit to phones"

    echo "computing the equivalent Phone Error Rate"
    python utils/ter.py --no-repeat $mapping $ref_ali \
        $hyp_ali > $outdir/ter || exit 1

    touch $outdir/.done_per
fi
echo "TER: $(tail -n 1 $outdir/ter)"

