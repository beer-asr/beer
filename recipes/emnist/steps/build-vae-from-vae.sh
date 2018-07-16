#!/bin/sh

usage() {
echo "Usage: $0 <conf-yamlfile> <dbstats> <init-vae> <outfile>"
}

help() {
echo "Create a VAE model from a configuration file and an existing VAE.
The encoder/decoder network of the new VAE will be initialized from
the existing VAE.
"
usage
echo "
Options:
  -h --help        show this message

Example:
  \$ $0 \\
        conf/vae.yml \\
        /path/to/dbstats.npz \\
        /path/to/existing-vae.py \\
        ./vae.mdl
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
if [ $# -ne 4 ]; then
    usage
    exit 1
fi

conf=$1
dbstats=$2
existing_vae=$3
model_output=$4

if [ ! -f "${model_output}" ]; then
    echo "Creating VAE from VAE..."
    python utils/create-vae-from-vae.py \
        "${conf}" \
        "${dbstats}" \
        "${existing_vae}" \
        "${model_output}" || exit 1
else
    echo "VAE already created. Skipping."
fi

