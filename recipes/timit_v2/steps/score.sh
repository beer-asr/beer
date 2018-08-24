#!/bin/bash

# Score all the transcription found ($expdir/*/decode/trans)


if [ $# -ne 1 ];then
    echo "$0 <setup.sh>"
    exit 1
fi

setup=$1
. $setup


rm -fr results.md

printf "| %-50s | %-20s |\n" "MODEL" "PER (%)" >> results.md
printf "|" >> results.md
printf '=%.0s' {1..52} | tr "=" "-">> results.md
printf "|" >> results.md
printf "=%.0s" {1..22} | tr "=" "-" >> results.md
printf "|\n" >> results.md

for trans in $(find $expdir -path '*/decode/trans') ; do
    mdl_dir=$(dirname $(dirname $trans))
    mdl_name=$(basename $mdl_dir)
    if [ ! -f $mdl_dir/decode/results.md ]; then
        #echo "Scoring $mdl_name ..."
        python utils/score.py \
            --remove=$remove_sym \
            --duplicate=$duplicate \
            --phone_map=$phone_48_to_39_map \
            $mdl_dir/decode/trans \
            $mdl_dir/decode/hyp \
                > $mdl_dir/decode/per || exit 1
    fi
    printf "| %-50s | %-20s |\n" \
        "[$mdl_name](conf/$mdl_name)" "$(cat $mdl_dir/decode/per)" \
        >> results.md

done

cat results.md

