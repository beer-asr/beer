
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import torch
from .bayesmodel import BayesianModel


class VAE(BayesianModel):
    '''Variational Auto-Encoder (VAE).'''

    def __init__(self, encoder, decoder, latent_model, nsamples):
        '''Initialize the VAE.

        Args:
            encoder (``MLPModel``): Encoder of the VAE.
            decoder (``MLPModel``): Decoder of the VAE.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
            nsamples (int): Number of samples to approximate the
                expectation of the log-likelihood.

        '''
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nsamples = nsamples

    @staticmethod
    def sufficient_statistics(data):
        return data

    def float(self):
        return self.__class__(
            self.encoder.float(),
            self.decoder.float(),
            self.latent_model.float(),
            self.nsamples
        )

    def double(self):
        return self.__class__(
            self.encoder.double(),
            self.decoder.double(),
            self.latent_model.double(),
            self.nsamples
        )

    def to(self, device):
        return self.__class__(
            self.encoder.to(device),
            self.decoder.to(device),
            self.latent_model.to(device),
            self.nsamples
        )

    def forward(self, s_stats, latent_variables=None):
        # For the case of the VAE, the sufficient statistics is just
        # the data itself. We just rename s_stats to avoid
        # confusion with the sufficient statistics of the latent model.
        data = s_stats

        enc_state = self.encoder(data)
        mean, var = enc_state.mean, enc_state.var
        exp_np_params, s_stats = self.latent_model.expected_natural_params(
            mean.detach(), var.detach(), latent_variables=latent_variables,
            nsamples=self.nsamples)
        self.cache['latent_stats'] = s_stats
        samples = mean + torch.sqrt(var) * torch.randn(self.nsamples,
                                                       data.size(0),
                                                       mean.size(1),
                                                       dtype=s_stats.dtype,
                                                       device=s_stats.device)
        self.cache['kl_divergence'] = enc_state.kl_div(exp_np_params)
        llh = self.decoder(samples).log_likelihood(data)
        return llh

    def local_kl_div_posterior_prior(self, parent_msg=None):
        return self.cache['kl_divergence'] + \
            self.latent_model.local_kl_div_posterior_prior()

    def accumulate(self, _, parent_msg=None):
        latent_stats = self.cache['latent_stats']
        self.clear_cache()
        return self.latent_model.accumulate(latent_stats, parent_msg)

__all__ = ['VAE']
