
# Directory structure.
confdir=conf
datadir=data
langdir=data/lang
timit=/export/corpora/LDC/LDC93S1/timit/TIMIT  # @JHU
#timit=/mnt/matylda2/data/TIMIT/timit  # @BUT
mdldir=exp/hmm_gmm

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
