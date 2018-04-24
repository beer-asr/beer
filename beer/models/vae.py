
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
        torch_data = Variable(torch.from_numpy(data).float())
        state = self(torch_data, sampling=sampling)
        loss, llh, kld = self.loss(torch_data, state)
        return -loss, llh, kld, state['encoder_state'].mean, \
            state['encoder_state'].std_dev ** 2

    def forward(self, X, sampling=True):
        '''Forward data through the VAE model.

        Args:
            x (torch.Variable): Data to process. The first dimension is
                the number of samples and the second dimension is the
                dimension of the latent space.
            sampling (boolean): If True, sample to approximate the
                expectation of the log-likelihood.

        Returns:
            dict: State of the VAE.

        '''
        # Forward the data through the encoder.
        encoder_state = self.encoder(X)

        # Forward the statistics to the latent model.
        p_np_params, acc_stats = self.latent_model.expected_natural_params(
            encoder_state.mean.data,
            encoder_state.var.data
        )

        # Samples of the latent variable using the reparameterization
        # "trick". "z" is a L x N x K tensor where L is the number of
        # samples for the reparameterization "trick", N is the number
        # of frames and K is the dimension of the latent space.
        if sampling:
            nsamples = self.nsamples
            samples = []
            for i in range(self.nsamples):
                samples.append(encoder_state.sample())
            samples = torch.stack(samples)#.view(self.nsamples * X.size(0), -1)
            decoder_state = self.decoder(samples)
        else:
            nsamples = 1
            decoder_state = self.decoder(encoder_state.mean)

        return {
            'encoder_state': encoder_state,
            'p_np_params': Variable(torch.FloatTensor(p_np_params)),
            'acc_stats': acc_stats,
            'decoder_state': decoder_state,
            'nsamples': nsamples
        }

    def loss(self, X, state, kl_weight=1.0):
        '''Loss function of the VAE. This is the negative of the
        variational objective function i.e.:

            loss = - ( E_q [ ln p(X|Z) ] - KL( q(z) || p(z) ) )

        Args:
            X (torch.Variable): Data on which to estimate the loss.
            state (dict): State of the VAE after forwarding the data
                through the network.
            kl_weight (float): Weight of the KL divergence in the loss.
                You probably don't want to touch it unless you know
                what you are doing.

        Returns:
            torch.Variable: Symbolic computation of the loss function.

        '''
        nsamples = state['nsamples']
        llh = state['decoder_state'].log_likelihood(X, state['nsamples'])
        llh = llh.view(nsamples, X.size(0), -1).sum(dim=0) / nsamples
        kl = state['encoder_state'].kl_div(state['p_np_params'])
        kl *= kl_weight

        return -(llh - kl[:, None]), llh, kl

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

    #def forward(self, T, labels=None):
    #    self.latent_model(T[1].data, labels.data)
    #    nparams = self.latent_model.components._expected_nparams_as_matrix().data
    #    nparams = Variable(self.latent_model._resps.data @ nparams)
    #    retval = self._state.log_likelihood(T[0]).mean(dim=0)
    #    retval -= self._state.kl_div(nparams)
    #    return retval

    def accumulate(self, T, parent_msg=None):
        return self.latent_model.accumulate(T[1].data, parent_msg)

