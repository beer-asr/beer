#!/bin/bash

if [ $# -ne 4 ];then
    echo "$0 <setup.sh> <model-dir> <data-dir> <decode-dir> "
    exit 1
fi

setup=$1
mdldir=$2
data_dir=$3
decode_dir=$4

. $setup
mdl=$mdldir/final.mdl
pdf_mapping=$mdldir/pdf_mapping.txt

[[ -f $pdf_mapping ]] || { echo "File not found: $pdf_mapping" >2; exit 1; }
mkdir -p $decode_dir

mdls=$(find $mdldir -name "[0-9]*mdl" -exec basename {} \; | \
    sort -t '.' -k 1 -g )

trap 'rm -rf "beer.*"' EXIT

for mdl in $mdls; do
    iter="${mdl%.*}"
    mkdir -p $decode_dir/iter$iter

    if [ ! -f $decode_dir/iter$iter/hyp ];then
        echo "Decoding with: $mdl"

# Load a numpy npz file and print its content as:
# utt1 pdf_id1 pdf_id2 ...
# utt2 pdf_id3 pdf_id4 ...
# ...
print_pdf_id="
import numpy as np
decoded = np.load('$decode_dir/iter$iter/best_paths.npz')
for utt in decoded.files:
    str_path = [str(pdf_id) for pdf_id in decoded[utt]]
    print(utt, ' '.join(str_path))
"

        tmpdir=$(mktemp -d $decode_dir/beer.XXXX);


        cmd="python utils/vae-hmm-decode-parallel.py $mdldir/$mdl \
            $data_dir/feats.npz $tmpdir"
        utils/parallel/submit_parallel.sh \
            "$parallel_env" \
            "hmm-decode" \
            "$hmm_decode_parallel_opts" \
            $hmm_decode_njobs \
            $data_dir/uttids \
            "$cmd" \
            $decode_dir/iter$iter || exit 1
        find $tmpdir -name '*npy' | zip -j -@ $decode_dir/iter$iter/best_paths.npz \
            > /dev/null || exit 1


        python -c "$print_pdf_id" | \
            python utils/pdf2unit.py --phone-level $pdf_mapping  \
            > $decode_dir/iter$iter/hyp
        ln -s $(pwd)/$data_dir/trans $decode_dir/iter$iter/trans
    else
        echo "Decoding already done. Skipping."
    fi
done


