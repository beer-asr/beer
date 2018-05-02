
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import abc
import math

import torch
from torch import nn
from torch.autograd import Variable

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

        # Temporary cache variable(s) used during training.
        self._T = None

    def sufficient_statistics(self, X):
        return X

    def forward(self, X, labels=None):
        enc_state = self.encoder(X)
        mean, var = enc_state.mean, enc_state.var
        self._T = self.latent_model.sufficient_statistics_from_mean_var(mean,
            var)
        exp_np_params = self.latent_model.expected_natural_params(
            mean.data, var.data, labels=labels, nsamples=self.nsamples)
        samples = mean + torch.sqrt(var) * torch.randn(self.nsamples, *X.size())
        llh = self.decoder(samples).log_likelihood(X)
        return llh - enc_state.kl_div(exp_np_params)

    def evaluate(self, X):
        'Convenience function mostly for plotting and debugging.'
        torch_data = Variable(torch.from_numpy(data).float())
        state = self(torch_data, sampling=sampling)
        loss, llh, kld = self.loss(torch_data, state)
        return -loss, llh, kld, state['encoder_state'].mean, \
            state['encoder_state'].std_dev ** 2

    def accumulate(self, _, parent_msg=None):
        return self.latent_model.accumulate(self._T, parent_msg)

