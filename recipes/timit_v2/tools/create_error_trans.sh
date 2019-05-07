#!/bin/bash

if [ $# != 5 ];then
    echo "$0: <srcdir> <tgtdir> <phones.txt> <num_wrong> <precision>"
    exit 1
fi

srcdir=$1
tgtdir=$2
phones=$3
num_wrong=$4
prec=$5

mkdir -p $tgtdir/tmp
uttids=$srcdir/uttids

# Create wrong subset
python utils/create_subset.py $srcdir $num_wrong $tgtdir/tmp
awk 'NR==FNR {a[$1];next} {if (!($1 in a)) print}' \
    $tgtdir/tmp/uttids \
    $srcdir/uttids > $tgtdir/tmp/seudo_true_uttids
awk 'NR==FNR {a[$1];next} $1 in a {print}' \
    $tgtdir/tmp/seudo_true_uttids \
    $srcdir/trans > $tgtdir/tmp/seudo_true_trans

# Create wrong trans
python tools/create_error_trans.py \
    $prec \
    $phones \
    $tgtdir/tmp/trans \
    $tgtdir/tmp/false_trans

cat $tgtdir/tmp/seudo_true_trans $tgtdir/tmp/false_trans > $tgtdir/tmp_trans
cp $srcdir/{uttids,feats.npz} $tgtdir
cut -d " " -f 1 $tgtdir/trans | sort > $tgtdir/uttids
diff $srcdir/trans $tgtdir/trans | grep "<" | \
    cut -d " " -f 2 | sort -u > $tgtdir/false_uttids
awk 'NR==FNR {a[$1];next} {if (!($1 in a)) print}' \
    $tgtdir/false_uttids \
    $srcdir/uttids > $tgtdir/true_uttids
awk 'NR==FNR {a[$1];next} $1 in a {print}' \
    $tgtdir/true_uttids \
    $srcdir/trans > $tgtdir/true_trans
awk 'NR==FNR {a[$1];next} $1 in a {print}' \
    $tgtdir/false_uttids \
    $srcdir/trans > $tgtdir/false_trans

new_num_wrong=$(wc -l $tgtdir/false_uttids | cut -d " " -f 1)
if [ $new_num_wrong != $num_wrong ]; then
    new_num_wrong="_${new_num_wrong}"
    echo $new_num_wrong
    #new_dir=$(echo $tgtdir | sed "s@(_.+)@_${num_wrong}@")
    echo $tgtdir | sed "s@(_.+)@$new_num_wrong@"
fi
#mv $tgtdir $new_dir
