
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import math
import torch
from .bayesmodel import BayesianModel


class VAE(BayesianModel):
    '''Variational Auto-Encoder (VAE).'''

    def __init__(self, encoder, encoder_problayer, decoder,  decoder_problayer,
                 latent_model):
        '''Initialize the VAE.

        Args:
            encoder (``torch.nn.Module``): Encoder of the VAE.
            encoder_problayer (:any:`beer.nnet.ProbabilisticLayer`):
                Layer to transform the output of the encoder into a
                probability distribution.
            decoder (``torch.nn.Module``): Decoder of the VAE.
            decoder_problayer (:any:`beer.nnet.ProbabilisticLayer`):
                Layer to transform the output of the decoder into a
                probability distribution.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
        '''
        super().__init__()
        self.encoder = encoder
        self.encoder_problayer = encoder_problayer
        self.decoder = decoder
        self.decoder_problayer = decoder_problayer
        self.latent_model = latent_model

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_model.mean_field_factorization()

    @staticmethod
    def sufficient_statistics(data):
        #For the VAE, this is just the idenity function
        return data

    def expected_log_likelihood(self, stats, kl_weight=1., use_mean=False,
                                **kwargs):
        encoder_states = self.encoder(stats)
        posterior_params = self.encoder_problayer(encoder_states)
        samples, post_llh = self.encoder_problayer.samples_and_llh(
                posterior_params, use_mean)

        # Per-frame KL divergence between the (approximate) posterior
        # and the prior.
        latent_stats = self.latent_model.sufficient_statistics(samples)
        prior_llh = self.latent_model.expected_log_likelihood(latent_stats,
                                                              **kwargs)
        kl_divs = post_llh - prior_llh

        # Log-likelihood.
        decoder_states = self.decoder(samples)
        llh_params = self.decoder_problayer(decoder_states)
        llhs = self.decoder_problayer.log_likelihood(stats, llh_params)

        # Store the statistics of the latent model to compute its
        # gradients
        self.cache['latent_stats'] = latent_stats

        return llhs - kl_weight * kl_divs

    def accumulate(self, _):
        latent_stats = self.cache['latent_stats']
        return self.latent_model.accumulate(latent_stats)


class VAEGlobalMeanVariance(VAE):
    '''Variational Auto-Encoder (VAE) with a global mean and
    (isostropic) covariance matrix parameters.

    '''

    def __init__(self, encoder, encoder_problayer, decoder,
                 normal, latent_model):
        super().__init__(encoder, encoder_problayer, decoder, None,
                         latent_model)
        self.normal = normal

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_model.mean_field_factorization() + \
            self.normal.mean_field_factorization()

    def expected_log_likelihood(self, data, kl_weight=1., use_mean=False,
                                **kwargs):
        encoder_states = self.encoder(data)
        posterior_params = self.encoder_problayer(encoder_states)
        samples, post_llh = self.encoder_problayer.samples_and_llh(
            posterior_params, use_mean)

        # Per-frame KL divergence between the (approximate) posterior
        # and the prior.
        latent_stats = self.latent_model.sufficient_statistics(samples)
        prior_llh = self.latent_model.expected_log_likelihood(latent_stats,
                                                              **kwargs)
        kl_divs = post_llh - prior_llh

        decoder_means = self.decoder(samples)
        centered_data = (data - decoder_means)
        centered_stats = self.normal.sufficient_statistics(centered_data)
        llhs = self.normal.expected_log_likelihood(centered_stats)

        # Store the statistics of the latent/likelihood model to
        # compute their gradients.
        self.cache['latent_stats'] = latent_stats.detach()
        self.cache['centered_stats'] = centered_stats.detach()
        return llhs - kl_weight * kl_divs

    def accumulate(self, _):
        latent_stats = self.cache['latent_stats']
        centered_stats = self.cache['centered_stats']
        return {
            **self.latent_model.accumulate(latent_stats),
            **self.normal.accumulate(centered_stats)
        }

__all__ = [
    'VAE',
    'VAEGlobalMeanVariance',
]
