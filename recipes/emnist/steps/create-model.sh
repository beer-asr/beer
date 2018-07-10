#!/bin/bash

usage() {
echo "Usage: $0 [options] <conf-yamlfile> <dbstats> <outfile>"
}

help() {
echo "Create a model from a configuration file."
echo ""
usage
echo ""
echo "Options:"
echo "  -h --help        show this message"
}

# Parsing optional arguments.
while [ $# -ge 0 ]; do
    param=`echo $1 | awk -F= '{print $1}'`
    value=`echo $1 | awk -F= '{print $2}'`
    case $param in
        -h | --help)
            help
            exit
            ;;
        --)
            shift
            break
            ;;
        -*)
            usage
            exit 1
            ;;
        *)
            break
    esac
    shift
done

# Parsing mandatory arguments.
if [ $# -ne 3 ]; then
    usage
    exit 1
fi

conf=$1
dbstats=$2
model_output=$3

if [ ! -f "${model_output}" ]; then
    echo "Creating model..."
    python utils/create-model.py \
        "${conf}" \
        "${dbstats}" \
        "${model_output}" || exit 1
else
    echo "Model already created. Skipping."
fi
