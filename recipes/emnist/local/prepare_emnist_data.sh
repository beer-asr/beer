#!/bin/bash

# Prepare the EMNIST dataset.

url=http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
batch_size=500
outdir=data

# Download the data
if [ ! -f "${outdir}/gzip.zip" ]; then
    mkdir -p ${outdir}
    wget $url -P "${outdir}" || exit 1
else
    echo "Data already downloaded. Skipping."
fi

# Extract and prepare the data.
if [ ! -f "${outdir}/.done" ]; then
    mkdir -p "${outdir}"/rawdata
    mkdir -p "${outdir}"/byclass/train "${outdir}"/byclass/test
    mkdir -p "${outdir}"/letters/train "${outdir}"/letters/test
    mkdir -p "${outdir}"/digits/train "${outdir}"/digits/test
    mkdir -p "${outdir}"/digits_{10,20,30,40,50,100,500,1000,5000,10000}/{train,test}
    mkdir -p "${outdir}/digits_nosampling/"{train,test}

    # Extract the "by-class" data set.
    unzip -p ${outdir}/gzip.zip gzip/emnist-byclass-train-images-idx3-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-byclass-train-images.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-byclass-train-labels-idx1-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-byclass-train-labels.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-byclass-test-images-idx3-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-byclass-test-images.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-byclass-test-labels-idx1-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-byclass-test-labels.raw" || exit 1

    #python local/raw2npz_byclass.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-byclass-train-images.raw" \
    #    "${outdir}/rawdata/emnist-byclass-train-labels.raw" \
    #    "${outdir}/byclass/train" || exit 1
    #find "${outdir}/byclass/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/byclass/train/archives" || exit 1

    #python local/raw2npz_byclass.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-byclass-train-images.raw" \
    #    "${outdir}/rawdata/emnist-byclass-train-labels.raw" \
    #    "${outdir}/byclass/test" || exit 1
    #find "${outdir}/byclass/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/byclass/test/archives" || exit 1

    ## Extract the letters data set.
    unzip -p ${outdir}/gzip.zip gzip/emnist-letters-train-images-idx3-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-letters-train-images.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-letters-train-labels-idx1-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-letters-train-labels.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-letters-test-images-idx3-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-letters-test-images.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-letters-test-labels-idx1-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-letters-test-labels.raw" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --label-offset -1 \
    #    "${outdir}/rawdata/emnist-letters-train-images.raw" \
    #    "${outdir}/rawdata/emnist-letters-train-labels.raw" \
    #    "${outdir}/letters/train" || exit 1
    #find "${outdir}/letters/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/letters/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --label-offset -1 \
    #    "${outdir}/rawdata/emnist-letters-train-images.raw" \
    #    "${outdir}/rawdata/emnist-letters-train-labels.raw" \
    #    "${outdir}/letters/test" || exit 1
    #find "${outdir}/letters/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/letters/test/archives" || exit 1

    ## Extract the digits data set.
    unzip -p ${outdir}/gzip.zip gzip/emnist-digits-train-images-idx3-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-digits-train-images.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-digits-train-labels-idx1-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-digits-train-labels.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-digits-test-images-idx3-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-digits-test-images.raw" || exit 1
    unzip -p ${outdir}/gzip.zip gzip/emnist-digits-test-labels-idx1-ubyte.gz | \
        gunzip > "${outdir}/rawdata/emnist-digits-test-labels.raw" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits/train" || exit 1
    #find "${outdir}/digits/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits/test" || exit 1
    #find "${outdir}/digits/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits/test/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --no-sampling \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits_nosampling/train" || exit 1
    #find "${outdir}/digits_nosampling/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_nosampling/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits_nosampling/test" || exit 1
    #find "${outdir}/digits_nosampling/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_nosampling/test/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        --train-nsamples 10 \
        "${outdir}/rawdata/emnist-digits-train-images.raw" \
        "${outdir}/rawdata/emnist-digits-train-labels.raw" \
        "${outdir}/digits_10/train" || exit 1
    find "${outdir}/digits_10/train" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_10/train/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        "${outdir}/rawdata/emnist-digits-test-images.raw" \
        "${outdir}/rawdata/emnist-digits-test-labels.raw" \
        "${outdir}/digits_10/test" || exit 1
    find "${outdir}/digits_10/test" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_10/test/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        --train-nsamples 20 \
        "${outdir}/rawdata/emnist-digits-train-images.raw" \
        "${outdir}/rawdata/emnist-digits-train-labels.raw" \
        "${outdir}/digits_20/train" || exit 1
    find "${outdir}/digits_20/train" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_20/train/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        "${outdir}/rawdata/emnist-digits-test-images.raw" \
        "${outdir}/rawdata/emnist-digits-test-labels.raw" \
        "${outdir}/digits_20/test" || exit 1
    find "${outdir}/digits_20/test" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_20/test/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        --train-nsamples 30 \
        "${outdir}/rawdata/emnist-digits-train-images.raw" \
        "${outdir}/rawdata/emnist-digits-train-labels.raw" \
        "${outdir}/digits_30/train" || exit 1
    find "${outdir}/digits_30/train" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_30/train/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        "${outdir}/rawdata/emnist-digits-test-images.raw" \
        "${outdir}/rawdata/emnist-digits-test-labels.raw" \
        "${outdir}/digits_30/test" || exit 1
    find "${outdir}/digits_30/test" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_30/test/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        --train-nsamples 40 \
        "${outdir}/rawdata/emnist-digits-train-images.raw" \
        "${outdir}/rawdata/emnist-digits-train-labels.raw" \
        "${outdir}/digits_40/train" || exit 1
    find "${outdir}/digits_40/train" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_40/train/archives" || exit 1

    python local/raw2npz.py \
        --batch-size ${batch_size} \
        "${outdir}/rawdata/emnist-digits-test-images.raw" \
        "${outdir}/rawdata/emnist-digits-test-labels.raw" \
        "${outdir}/digits_40/test" || exit 1
    find "${outdir}/digits_40/test" \
        -name 'batch*npz' | sort -V > "${outdir}/digits_40/test/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --train-nsamples 50 \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits_50/train" || exit 1
    #find "${outdir}/digits_50/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_50/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits_50/test" || exit 1
    #find "${outdir}/digits_50/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_50/test/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --train-nsamples 100 \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits_100/train" || exit 1
    #find "${outdir}/digits_100/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_100/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits_100/test" || exit 1
    #find "${outdir}/digits_100/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_100/test/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --train-nsamples 500 \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits_500/train" || exit 1
    #find "${outdir}/digits_500/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_500/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits_500/test" || exit 1
    #find "${outdir}/digits_500/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_500/test/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --train-nsamples 1000 \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits_1000/train" || exit 1
    #find "${outdir}/digits_1000/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_1000/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits_1000/test" || exit 1
    #find "${outdir}/digits_1000/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_1000/test/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --train-nsamples 5000 \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits_5000/train" || exit 1
    #find "${outdir}/digits_5000/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_5000/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits_5000/test" || exit 1
    #find "${outdir}/digits_5000/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_5000/test/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    --train-nsamples 10000 \
    #    "${outdir}/rawdata/emnist-digits-train-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-train-labels.raw" \
    #    "${outdir}/digits_10000/train" || exit 1
    #find "${outdir}/digits_10000/train" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_10000/train/archives" || exit 1

    #python local/raw2npz.py \
    #    --batch-size ${batch_size} \
    #    "${outdir}/rawdata/emnist-digits-test-images.raw" \
    #    "${outdir}/rawdata/emnist-digits-test-labels.raw" \
    #    "${outdir}/digits_10000/test" || exit 1
    #find "${outdir}/digits_10000/test" \
    #    -name 'batch*npz' | sort -V > "${outdir}/digits_10000/test/archives" || exit 1


    rm -fr "${outdir}/rawdata" || exit 1

    date > "${outdir}/.done"
else
    echo "Data already prepared. Skipping."
fi
