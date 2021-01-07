#!/bin/bash

set -e
rootdir=/mnt/matylda2/data/google_lr
#rootdir=$(realpath ~/google_lr/)
if [ $# -ne 2 ]; then
    echo "$0 <out-dir> <lang>"
    exit 1;
fi

outdir=$1
languages=$2


for language in $languages; do
    cmddir=$(dirname $0)
    for subset in train; do
	datadir=$outdir/$language
	[ -f $outdir/$language/$subset/uttids ] && exit 0
	echo $outdir/$language/$subset/uttids
       	mkdir -p $outdir/$language/$subset
	find -L $rootdir/$language -name "*.wav" | \
	    awk -F . '{split($1, a, "/"); print a[length(a)], $0}' \
		> $datadir/$subset/all_wavs.txt
	cat $rootdir/$language/*/line_index.tsv > $datadir/$subset/all_trans.txt

	python3 $cmddir/prepare_transcripts.py \
		--add-sil \
		--lang ${language}_tones \
		--diacritics \
		$datadir/$subset/all_trans.txt \
		$outdir/${language}/$subset

	awk 'NR==FNR {a[$1] = $2}; NR!=FNR{print $1, a[$1]'} \
	    $datadir/$subset/all_wavs.txt \
	    $outdir/$language/$subset/trans \
	    > $outdir/$language/$subset/wav.scp
	cut -d ' ' -f1 $outdir/$language/$subset/trans > $outdir/$language/$subset/uttids
    done
    echo "sil non-speech-unit" > $outdir/$language/lang/units
    cut -d' ' -f2- $outdir/$language/train/trans | \
	tr ' ' '\n' | \
	sort -u | grep -v sil | \
	awk '{print $0, "speech-unit"}' >> $outdir/$language/lang/units
    [ -f $cmddir/ali ] && cp $cmddir/ali $outdir/$language/train/
done
