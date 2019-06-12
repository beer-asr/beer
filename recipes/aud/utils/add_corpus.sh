#!/usr/bin/env bash

max_utts=1000000
nargs=2
while [[ $# -gt $nargs ]]; do
    case $1 in
      --max-utts)
      max_utts=$2
      shift
      shift
      ;;
      *)
      echo "unknown option: $1"
      exit 1
    esac
done

if [ $# -ne $nargs ]; then
    echo "usage: $0 [OPTS] <data-dir> <out-dir>"
    echo ""
    echo "Add a corpus"
    echo ""
    echo "Options:"
    echo "  --max-utts          maximum number of utterances to keep"
    echo ""
    exit 1
fi

corpus=$1
outdir=$2


for x in train dev test; do
    mkdir -p $outdir/$x
    cat $corpus/$x/uttids | shuf | head -n $max_utts >> $outdir/$x/uttids
    cat $corpus/$x/wav.scp | grep -w -f $outdir/$x/uttids >> $outdir/$x/wav.scp
    cat $corpus/$x/trans | grep -w -f $outdir/$x/uttids >> $outdir/$x/trans
done

mkdir -p $outdir/lang
cat $corpus/lang/units >> $outdir/lang/units
mv $outdir/lang/units $outdir/lang/units.bk
cat $outdir/lang/units.bk | sort | uniq > $outdir/lang/units

