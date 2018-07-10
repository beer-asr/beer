#!/bin/sh

gpu=  # Empty variable means we don't use the GPU.
nsamples=5

usage() {
echo "Usage: $0 [--use-gpu [--nsamples=N]] <sge-options> <model> <archives> <outdir>"
}

help() {
echo "Compute the classification accuracy of a Variational Auto-Encoder"
echo "with discrete latent model prior (i.e. GMM or similar)."
echo ""
usage
echo ""
echo "Options:"
echo "  -h --help        show this message"
echo "  --use-gpu        use the gpu"
echo "  --nsamples       number of samples for the re-parameterization trick"
echo ""
echo "Example:"
echo "  \$ $0 \\
            --nsamples=5 -- \\
            \"-l mem_free=1G,ram_free=1G\" \\
            /path/to/model.mdl \\
            /path/to/archives/ \\
            results_dir"
echo ""
echo "Note the double hyphens \"--\" to avoid problem when parsing"
echo "the SGE option \"-l ...\"."
echo ""
}

# Parsing optional arguments.
while [ $# -ge 0 ]; do
    param=$(echo $1 | awk -F= '{print $1}')
    optname=$(echo ${param} | sed 's/--//g' | sed 's/-/_/g')
    value=`echo $1 | awk -F= '{print $2}'`
    case $param in
        -h | --help)
            help
            exit
            ;;
        --use-gpu)
            gpu="--use-gpu"
            shift
            ;;
        --nsamples)
            eval ${optname}=${value}
            shift
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
done

# Parsing mandatory arguments.
if [ $# -ne 4 ]; then
    usage
    exit 1
fi

sge_options=$1
model=$2
archives=$3
outdir=$4


mkdir -p "${outdir}"


options="\
${gpu}  \
--nsamples ${nsamples} \
"

if [ ! -f "${outdir}/.done" ]; then
    echo "Computing accuracy..."

    # Command to submit to the SGE.
    cmd="python utils/accuracy-vae-discrete-latent-model.py \
        ${options} \
        ${archives} \
        ${model} \
        ${outdir}/accuracy"

    # Clear the log file.
    rm -f ${outdir}/sge.log

    # Submit the command to the SGE.
    qsub \
        ${sge_options} \
        -cwd \
        -j y \
        -sync y \
        -o ${outdir}/sge.log \
        utils/job.qsub \
        "${cmd}" || exit 1

    date > "${outdir}/.done"
else
    echo "Accuracy already computed. Skipping."
fi

cat "${outdir}/accuracy"

