#!/bin/bash


input_trans="data/test_eval92/text"
output_trans="data/test_eval92_corrupted/text"
sub_list="data/test_eval92_corrupted/substitute_words.txt"
ins_list="data/test_eval92_corrupted/ins.list"
del_ins_true_rate="0.1052 0.0526 0.8422"
ser=0.4

python tools/data_prep/create_corrupted_trans.py \
    --del_ins_true_rate "$del_ins_true_rate" \
    --ser "$ser" \
    "$sub_list" \
    "$ins_list" \
    "$input_trans" \
    "$output_trans"


indir=`dirname $input_trans`
outdir=`dirname $output_trans`

cat $output_trans | local/wer_hyp_filter > $outdir/text_filt.txt
compute-wer --text --mode=present ark:"$indir/text_filt.txt" \
    ark:"$outdir/text_filt.txt" \
    > $outdir/wer_details

diff $indir/text_filt.txt $outdir/text_filt.txt | grep "<" | \
    cut -d " " -f2 > $outdir/false_uttids

awk 'NR==FNR{a[$1];next}{if (!($1 in a)) print $1}' $outdir/false_uttids \
    $outdir/utt2spk > $outdir/true_uttids

