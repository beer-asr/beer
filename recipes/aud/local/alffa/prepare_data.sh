#!/bin/bash

set -e
rootdir=/mnt/matylda2/data/ALFFA

if [ $# -ne 2 ]; then
    echo "$0 <out-dir> <lang>"
    exit 1;
fi

outdir=$1
languages=$2

for language in $languages; do
    datadir=$rootdir/data_readspeech_$language
    [ ! -d $datadir ] && datadir=$rootdir/data_broadcastnews_$language
    [ ! -d $datadir ] && echo "datadir $datadir does not exist" && exit 1
    lexicon=$datadir/lang/lexicon.txt
    [ ! -f $lexicon ] && lexicon=$datadir/lang_o3g_LengthContrast/lexicon.txt
    [ ! -f $lexicon ] && echo "lexicon $lexicon not found" && exit 1
    cmddir=$(dirname $0)

    for subset in train test; do
	[ -f $outdir/$language/$subset/uttids ] && continue
       	mkdir -p $outdir/$language/$subset/tmp
	find $datadir/data/$subset/ -name "*.wav" | \
	    awk -F . '{split($1, a, "/"); print a[length(a)], $0}' \
		> $outdir/$language/$subset/tmp/all_wavs.txt

	python3 $cmddir/prepare_transcripts.py \
		--lang $language \
		$datadir/data/$subset/text \
		$lexicon \
		$outdir/$language/$subset
	awk 'NR==FNR {a[$1] = $0}; NR!=FNR{print a[$1]}' \
	    $outdir/$language/$subset/tmp/all_wavs.txt \
	    $outdir/$language/$subset/trans \
	    > $outdir/$language/$subset/wav.scp
	cut -d ' ' -f1 $outdir/$language/$subset/trans > $outdir/$language/$subset/uttids

    rm -fr $outdir/$language/$subset/tmp

    done
    echo "sil non-speech-unit" > $outdir/$language/lang/units
    cut -d' ' -f2- $outdir/$language/train/trans | \
	tr ' ' '\n' | \
	sort -u | grep -v sil | \
	awk '{print $0, "speech-unit"}' >> $outdir/$language/lang/units
done
