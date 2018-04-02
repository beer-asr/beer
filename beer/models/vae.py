
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
        self._state = None

    def evaluate(self, X):
        'Convenience function mostly for plotting and debugging.'
        state = self.encoder(X)
        return state.mean, state.var

    def sufficient_statistics(self, X):
        self._state = self.encoder(X)
        nsamples = self.nsamples
        samples = []
        for i in range(self.nsamples):
            samples.append(self._state.sample())
        samples = torch.stack(samples)
        #.view(self.nsamples * X.size(0), -1)
        T = self.latent_model.sufficient_statistics(samples), \
            self.latent_model.sufficient_statistics_from_mean_var(
                self._state.mean, self._state.var)
        return T

    def forward(self, T, labels=None):
        self.latent_model(T[1].data, labels.data)
        nparams = self.latent_model._components._expected_nparams_as_matrix().data
        nparams = Variable(self.latent_model._resps.data @ nparams)
        retval = self._state.log_likelihood(T[0]).mean(dim=0)
        retval -= self._state.kl_div(nparams)
        return retval

    def accumulate(self, T, parent_msg=None):
        return self.latent_model.accumulate(T[1].data, parent_msg)

