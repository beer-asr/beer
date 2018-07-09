#!/bin/bash

help="\
Train  a VAE model.

usage: $0 <init-model> <dbstats> <archives-list> <out-model>
"

if [ $# -ne 4 ]; then
    echo "${help}"
    exit 1
fi