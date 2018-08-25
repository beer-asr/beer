#!/bin/bash


if [ $# -ne 2 ];then
    echo "$0 <setup.sh> <decode-dir>"
    exit 1
fi

setup=$1
decode_dir=$2
. $setup


rm -fr results.md

printf "| %-10s | %-10s |\n" "ITERATION" "PER (%)"
printf "|"
printf '=%.0s' {1..12} | tr "=" "-"
printf "|"
printf "=%.0s" {1..12} | tr "=" "-"
printf "|\n"

for iter in $(find $decode_dir -name 'iter*' -exec basename {} \; | sed 's/iter//g' ) ; do
    if [ ! -f $decode_dir/iter$iter/per ]; then
        python utils/score.py \
            --remove=$remove_sym \
            --duplicate=$duplicate \
            --phone_map=$phone_48_to_39_map \
            $decode_dir/iter$iter/trans \
            $decode_dir/iter$iter/hyp > $decode_dir/iter$iter/per || exit 1
    fi
    printf "| %-10s | %-10s |\n" "$iter" "$(cat $decode_dir/iter$iter/per)"
done

