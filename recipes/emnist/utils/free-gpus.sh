#!/bin/sh

usage() {
echo "Usage: $0 [-h | --help]"
}

help() {
echo "Get the list of the available of available gpus.
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
if [ $# -ne 0 ]; then
    usage
    exit 1
fi


# Get the number of GPU available. If the `nvidia-smi` command fails
# we set this number to 0.
nvidia_smi_output=$(nvidia-smi -L)
if [ ${?} -eq 0 ]; then
    n_gpus=$(echo "${nvidia_smi_output}" | wc -l)
else
    n_gpus=0
fi


if [ ${n_gpus} -eq 0 ]; then
    echo "No GPUs on this machine." 1>&2
    exit 1
fi

for i in $(seq 1 $n_gpus); do
    gpu_id=$((i-1))
    status=`nvidia-smi -i ${gpu_id} | grep -c "No running processes found"`
    if [ "$status" = "1" ]; then
        if [ ! -z ${free_gpus} ]; then
            free_gpus=${free_gpus},
        fi
        free_gpus=${free_gpus}${gpu_id}
    fi
done

echo ${free_gpus}

