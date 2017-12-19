
"""Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

"""

import math

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np
from .model import Model


class VAE(nn.Module, Model):
    """Variational Auto-Encoder (VAE)."""


    def __init__(self, encoder, decoder, latent_model, nsamples):
        """Initialize the VAE.

        Args:
            encoder (``NormalMLP``): Encoder of the VAE.
            decoder (``NormalMLP``): Decoder of the VAE.
            latent_model(``Model``): Bayesian Model for the prior over
                the latent space.
            nsamples (int): Number of samples to approximate the
                expectation of the log-likelihood.

        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nsamples = nsamples

    def _fit_step(self, mini_batch):
        mini_batch_size = np.prod(mini_batch.shape[:-1])
        data_size = self._fit_cache['data_size']
        scale = float(data_size) / mini_batch_size
        X = Variable(torch.from_numpy(mini_batch).float())
        self._fit_cache['optimizer'].zero_grad()
        state = self(X)
        loss, llh, kld = self.loss(X, state,
            kl_weight=self._fit_cache['kl_weight'])
        loss, llh, kld = loss.sum(), llh.sum(), kld.sum()
        loss /= X.size(0)
        loss.backward()
        self._fit_cache['optimizer'].step()
        self.latent_model.natural_grad_update(state['acc_stats'],
            scale=scale, lrate=self._fit_cache['latent_model_lrate'])

        elbo  =-loss.data.numpy()[0] \
            - self.latent_model.kl_div_posterior_prior() / data_size

        return elbo, llh.data.numpy()[0], kld.data.numpy()[0]

    def fit(self, data, mini_batch_size=-1, max_epochs=1, seed=None, lrate=1e-3,
            latent_model_lrate=1., kl_weight=1.0, callback=None):
        self._fit_cache = {
            'optimizer':optim.Adam(self.encoder.parameters(), lr=lrate,
                                   weight_decay=1e-6),
            'latent_model_lrate': latent_model_lrate,
            'data_size': np.prod(data.shape[:-1]),
            'kl_weight': kl_weight
        }
        super().fit(data, mini_batch_size, max_epochs, seed, callback)

    def evaluate(self, data, sampling=True):
        torch_data = Variable(torch.from_numpy(data).float())
        state = self(torch_data, sampling=sampling)
        loss, llh, kld = self.loss(torch_data, state)
        return -loss.data.numpy(), llh.data.numpy(), kld.data.numpy(), \
            state['encoder_state']['mean'].data.numpy(), \
            torch.pow(state['encoder_state']['std_dev'], 2).data.numpy(), \
            state['decoder_state']['mean'].data.numpy(), \
            torch.pow(state['decoder_state']['std_dev'], 2).data.numpy()


    def forward(self, x, sampling=True):
        """Forward data through the VAE model.

        Args:
            x (torch.Variable): Data to process. The first dimension is
                the number of samples and the second dimension is the
                dimension of the latent space.
            sampling (boolearn): If True, sample to approximate the
                expectation of the log-likelihood.

        Returns:
            dict: State of the VAE.

        """
        encoder_state = self.encoder(x)
        # Forward the statistics to the latent model.
        p_np_params, acc_stats = self.latent_model.expected_natural_params(
                encoder_state['mean'].data.numpy(),
                (encoder_state['std_dev']**2).data.numpy())

        # Samples of the latent variable using the reparameterization
        # "trick". "z" is a L x N x K tensor where L is the number of
        # samples for the reparameterization "trick", N is the number
        # of frames and K is the dimension of the latent space.
        if sampling:
            samples = []
            for i in range(self.nsamples):
                samples.append(self.encoder.sample(encoder_state))
            samples = torch.stack(samples)
            decoder_state = self.decoder(samples)
        else:
            decoder_state = self.decoder(encoder_state['mean'])

        return {
            'encoder_state': encoder_state,
            'p_np_params': Variable(torch.FloatTensor(p_np_params)),
            'acc_stats': acc_stats,
            'decoder_state': decoder_state
        }

    def loss(self, x, state, kl_weight=1.0):
        """Loss function of the VAE. This is the negative of the
        variational objective function i.e.:

            loss = - ( E_q [ ln p(X|Z) ] - KL( q(z) || p(z) ) )

        Args:
            x (torch.Variable): Data on which to estimate the loss.
            state (dict): State of the VAE after forwarding the data
                through the network.

        Returns:
            torch.Variable: Symbolic computation of the loss function.

        """
        llh = self.decoder.log_likelihood(x, state['decoder_state'])
        kl = self.encoder.kl_div(state['encoder_state'], state['p_np_params'])
        kl *= kl_weight

        return -(llh - kl), llh, kl


class MLPNormalDiag(nn.Module):
    '''Neural-Network ending with a double linear projection
    providing the mean and the logarithm of the diagonal of the
    covariance matrix.

    '''
    @staticmethod
    def sufficient_statistics(data):
        '''Compute the sufficient statistics of the data.'''
        return torch.cat([data**2, data,
                          Variable(torch.ones(data.size()).float()),
                          Variable(torch.ones(data.size()).float())], dim=1)

    @staticmethod
    def log_likelihood(data, state):
        '''Log-likelihood of the data give the current state of the
        network.

        '''
        s_stats = MLPNormalDiag.sufficient_statistics(data)
        nparams = state['natural_params']
        log_base_measure = -.5 * data.size(1) * math.log(2 * math.pi)
        llh = torch.sum(nparams * s_stats, dim=-1)
        if len(llh.size()) > 1: llh = llh.sum(dim=0) / nparams.size(0)
        llh += log_base_measure

        return llh

    @staticmethod
    def sample(state):
        '''Sample data using the reparametization trick.'''
        mean = state['mean']
        std_dev = state['std_dev']
        noise = Variable(torch.randn(*mean.size()))
        return mean + std_dev * noise

    @staticmethod
    def kl_div(state, p_nparams):
        '''KL divergence between the posterior distribution and the
        prior.

        '''
        return ((state['natural_params'] - p_nparams) * state['exp_T']).sum(dim=-1)

    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logprec = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        if len(x.size()) > 2:
            x_reshaped = x.view(-1, x.size(2))
        else:
            x_reshaped = x

        h = self.structure(x_reshaped)
        mean = self.hid_to_mu(h).view(*x.size())
        logprec = self.hid_to_logprec(h)
        diag_prec = torch.exp(logprec.view(*x.size()))
        return {
            'exp_T': torch.cat([mean**2 +  1/diag_prec, mean,
                                Variable(torch.ones(*x.size())),
                                Variable(torch.ones(*x.size()))],
                               dim=1),
            'mean': mean.view(*x.size()),
            'std_dev': torch.exp(-.5 * logprec).view(*x.size()),
            'natural_params': torch.cat([
                (-.5 * diag_prec).view(*x.size()),
                (diag_prec * mean).view(*x.size()),
                (-.5 * diag_prec * (mean ** 2)).view(*x.size()),
                (.5 * logprec).view(*x.size())
            ], dim=-1)
        }

class MLPNormalDiag2(nn.Module):
    '''Neural-Network ending with a double linear projection
    providing the mean and the logarithm of the diagonal of the
    covariance matrix.

    '''
    @staticmethod
    def sufficient_statistics(data):
        '''Compute the sufficient statistics of the data.'''
        data2 = data[:, :, None] * data[:, None, :]
        return torch.cat([data2.view(data.size(0), -1), data,
                          Variable(torch.ones(data.size(0)).float()),
                          Variable(torch.ones(data.size(0)).float())], dim=1)

    @staticmethod
    def log_likelihood(data, state):
        '''Log-likelihood of the data give the current state of the
        network.

        '''
        s_stats = MLPNormalDiag.sufficient_statistics(data)
        nparams = state['natural_params']
        log_base_measure = -.5 * data.size(1) * math.log(2 * math.pi)
        llh = torch.sum(nparams * s_stats, dim=-1)
        if len(llh.size()) > 1: llh = llh.sum(dim=0) / nparams.size(0)
        llh += log_base_measure

        return llh

    @staticmethod
    def sample(state):
        '''Sample data using the reparametization trick.'''
        mean = state['mean']
        std_dev = state['std_dev']
        noise = Variable(torch.randn(*mean.size()))
        return mean + std_dev * noise

    @staticmethod
    def kl_div(state, p_nparams):
        '''KL divergence between the posterior distribution and the
        prior.

        '''
        return ((state['natural_params'] - p_nparams) * state['exp_T']).sum(dim=-1)

    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logprec = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        if len(x.size()) > 2:
            x_reshaped = x.view(-1, x.size(2))
        else:
            x_reshaped = x

        h = self.structure(x_reshaped)
        mean = self.hid_to_mu(h).view(*x.size())
        logprec = self.hid_to_logprec(h)
        diag_prec = torch.exp(logprec.view(*x.size()))

        # Expected value of the sufficient statistics.
        idxs = torch.arange(0, x_reshaped.size(1)).long()
        mean2 = mean[:, :, None] * mean[:, None, :]
        XX = mean2
        XX[:, idxs, idxs] += 1 / diag_prec

        # First natural parameter (-.5 * precision_matrix).
        identity = Variable(torch.eye(x_reshaped.size(1)))
        np1 = -.5 * torch.exp(logprec)[:, None] * identity[None, :, :]
        np1 = np1.view(logprec.size(0), -1)

        np = torch.cat([
                np1,
                (diag_prec * mean).view(*x.size()),
                -.5 * (diag_prec * mean ** 2).sum(-1)[:, None],
                (.5 * logprec).sum(-1)[:, None]
        ], dim=-1)

        return {
            'exp_T': torch.cat([XX.view(mean.size(0), -1), mean,
                                Variable(torch.ones(x.size(0), 1)),
                                Variable(torch.ones(x.size(0), 1))], dim=1),
            'mean': mean.view(*x.size()),
            'std_dev': torch.exp(-.5 * logprec).view(*x.size()),
            'natural_params': np
        }

class MLPNormalIso(nn.Module):
    '''Neural-Network ending with a double linear projection
    providing the mean and the isotropic covariance matirx

    '''
    @staticmethod
    def sufficient_statistics(data):
        '''Compute the sufficient statistics of the data.'''
        data2 = data[:, :, None] * data[:, None, :]
        return torch.cat([data2.view(data.size(0), -1), data,
                          Variable(torch.ones(data.size(0)).float()),
                          Variable(torch.ones(data.size(0)).float())], dim=1)

    @staticmethod
    def log_likelihood(data, state):
        '''Log-likelihood of the data give the current state of the
        network.

        '''
        s_stats = MLPNormalIso.sufficient_statistics(data)
        nparams = state['natural_params']
        log_base_measure = -.5 * data.size(1) * math.log(2 * math.pi)
        llh = torch.sum(nparams * s_stats, dim=-1)
        if len(llh.size()) > 1: llh = llh.sum(dim=0) / nparams.size(0)
        llh += log_base_measure

        return llh

    @staticmethod
    def sample(state):
        '''Sample data using the reparametization trick.'''
        mean = state['mean']
        std_dev = state['std_dev']
        noise = Variable(torch.randn(*mean.size()))
        return mean + std_dev * noise

    @staticmethod
    def kl_div(state, p_nparams):
        '''KL divergence between the posterior distribution and the
        prior.

        '''
        return ((state['natural_params'] - p_nparams) * state['exp_T']).sum(dim=-1)

    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logprec = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if len(x.size()) > 2:
            x_reshaped = x.view(-1, x.size(2))
        else:
            x_reshaped = x

        # Forward the data through the neural network.
        h = self.structure(x_reshaped)

        # Get the mean and the diagonal of the precision matrix.
        mean = self.hid_to_mu(h).view(*x.size()) + x
        logprec = self.hid_to_logprec(h)
        diag_prec = Variable(torch.ones(mean.size(1))) * torch.exp(logprec)
        diag_prec = diag_prec.view(*x.size())

        # Expected value of the sufficient statistics.
        idxs = torch.arange(0, x_reshaped.size(1)).long()
        mean2 = mean[:, :, None] * mean[:, None, :]
        XX = mean2
        XX[:, idxs, idxs] += 1/ diag_prec

        # First natural parameter (-.5 * precision_matrix).
        identity = Variable(torch.eye(x_reshaped.size(1)))
        np1 = -.5 * torch.exp(logprec)[:, None] * identity[None, :, :]
        np1 = np1.view(logprec.size(0), -1)

        np = torch.cat([
                np1,
                (diag_prec * mean).view(*x.size()),
                -.5 * (diag_prec * mean ** 2).sum(1)[:, None],
                (.5 * mean.size(1) * logprec)
        ], dim=-1)

        return {
            'exp_T': torch.cat([XX.view(mean.size(0), -1), mean,
                                Variable(torch.ones(x.size(0), 1)),
                                Variable(torch.ones(x.size(0), 1))], dim=1),
            'mean': mean.view(*x.size()),
            'std_dev': torch.sqrt(1/diag_prec).view(*x.size()),
            'natural_params': np
        }
