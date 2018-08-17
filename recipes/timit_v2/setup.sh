
# Directory structure.
timit=/export/corpora/LDC/LDC93S1/timit/TIMIT  # @JHU
#timit=/mnt/matylda2/data/TIMIT/timit  # @BUT
confdir=$(pwd)/conf
datadir=$(pwd)/data
data_train_dir=$(pwd)/data/train
langdir=$(pwd)/data/lang
root=$(pwd)/exp


# Features extraction.
fea_njobs=10
fea_sge_opts="-l mem_free=100M,ram_free=100M"
fea_conf=$confdir/features.yml


# VAE-HMM model.
vae_hmm_confdir=$(pwd)/conf/vae_hmm
vae_hmm_encoder_conf=$vae_hmm_confdir/encoder.yml
vae_hmm_decoder_conf=$vae_hmm_confdir/decoder.yml
vae_hmm_normalizing_flow_conf=$vae_hmm_confdir/normalizing_flow.yml
vae_hmm_emissions_conf=$vae_hmm_confdir/emissions.yml
vae_hmm_latent_dim=30
vae_hmm_encoder_out_dim=128
vae_hmm_encoder_cov_type=isotropic
vae_hmm_decoder_cov_type=diagonal
vae_hmm_training_type=viterbi
vae_hmm_lrate=1e-1
vae_hmm_lrate_nnet=1e-3
vae_hmm_batch_size=400
vae_hmm_epochs=10
vae_hmm_opts="--fast-eval --use-gpu"
vae_hmm_dir=$root/vae_hmm_nw${vae_hmm_encoder_out_dim}_ldim${vae_hmm_latent_dim}


# HMM-GMM model parameters.
hmm_gmm_mdl_dir=exp/hmm_gmm
hmm_conf=$confdir/hmm_gmm.yml
hmm_infer_type='viterbi'
hmm_lrate=0.1
hmm_batch_size=400
hmm_epochs=10
hmm_fast_eval="--fast-eval"
use_gpu=""
