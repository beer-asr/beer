#!/usr/bin/env bash

. path.sh

expdir=exp
nargs=1

while [[ $# -gt $nargs ]]; do
    case $1 in
      --exp-dir)
      expdir=$2
      shift
      shift
      ;;
      *)
      echo "unknown option: $1"
      exit 1
    esac
done

if [ $# -ne $nargs ]; then
    echo "usage: $0 [OPTS] <database>"
    echo ""
    echo "Show a summary of the results"
    echo ""
    echo "Options:"
    echo "  --exp-dir           experiment directory (default: exp)"
    echo ""
    exit 1
fi

db=$1

header="$(printf "| %-70s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s |" model dataset precision recall f-score perplexity "NMI")"
len=$(($(echo "$header" | wc -c) -1 ))
printf '=%.0s' $(eval echo "{1..$len}")
printf '\n'
printf "$header\n"
printf '=%.0s' $(eval echo "{1..$len}")
printf '\n'

for path in "$expdir/$db"/*/decode_perframe/*/score; do
    model=$(basename $(dirname $(dirname $(dirname $path))))
    dataset=$(basename $(dirname $path))
    nmi=$(tail -n1 $path/nmi)
    prec=$(tail -n1 $path/phone_boundaries | cut -d, -f1)
    rec=$(tail -n1 $path/phone_boundaries | cut -d, -f2)
    fscore=$(tail -n1 $path/phone_boundaries | cut -d, -f3)
    perplexity=$(tail -n1 $path/entropy_rate| cut -d, -f2)
    printf "| %-70s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s |\n" $model $dataset $prec $rec $fscore $perplexity $nmi
done
printf '=%.0s' $(eval echo "{1..$len}")
printf '\n'

