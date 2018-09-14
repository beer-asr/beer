rootdir=$(pwd)
modeldir=$rootdir/exp/vae_hmm_per_utt_norm
datadir=$rootdir/data/test_50_wrong_trans
outdir=$rootdir/icassp_task/vae_hmm/test_50_wrong_trans
alidir=$rootdir/exp/vae_hmm_per_utt_norm/ali_test_50_wrong_trans
level="frame"
thres="0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,40,50,60"

## Parallel opts
parallel_env=sge
parallel_opts="-l mem_free=200M,ram_free=200M,hostname=b*|c*"
njobs=20

## Align opts
hmm_align_parallel_opts=$parallel_opts
hmm_align_njobs=20

