#!/bin/sh

usage() {
echo "Usage: $0 <archives-list> <out-npzfile>"
}

help() {
echo "Compute the statistics (mean/var/counts) of a database.
"
usage
echo "
Options:
  -h --help        show this message

Example:
  \$ $0 \\
        /path/to/archives_list \\
       ./dbstats.npz
"
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
if [ $# -ne 2 ]; then
    usage
    exit 1
fi

archives=$1
outfile=$2

if [ ! -f "${outfile}" ]; then
    echo "Computing stats..."
    python utils/compute-stats.py ${archives} ${outfile} || exit 1
else
    echo "Statistics already computed. Skipping."
fi

