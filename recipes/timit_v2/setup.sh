

# Directory structure.
confdir=$(pwd)/conf
datadir=$(pwd)/data
timit=/export/corpora/LDC/LDC93S1/timit/TIMIT  # @JHU
#timit=/mnt/matylda2/data/TIMIT/timit  # @BUT

# Features extraction.
fea_njobs=10
fea_sge_opts="-l mem_free=100M,ram_free=100M"
fea_conf=$confdir/features.yml

