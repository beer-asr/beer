
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
        self.sample = True

    def _fit_step(self, mini_batch):
        X = Variable(torch.from_numpy(mini_batch).float())
        self._fit_cache['optimizer'].zero_grad()
        state = self(X)
        loss, llh, kld = self.loss(X, state)
        loss.backward()
        self._fit_cache['optimizer'].step()

        return -loss.data.numpy(), llh.data.numpy(), kld.data.numpy()

    def fit(self, data, mini_batch_size=-1, max_epochs=1, seed=None, lrate=1e-3,
            latent_model_lrate=1.):
        self._fit_cache = {
            'optimizer':optim.Adam(self.parameters(), lr=lrate, 
                                   weight_decay=1e-6),
            'latent_model_lrate': latent_model_lrate,
            'data_size': np.prod(data.shape[:-1])
        }
        super().fit(data, mini_batch_size, max_epochs,seed)


    def forward(self, x):
        """Forward data through the VAE model.

        Args:
            x (torch.Variable): Data to process. The first dimension is
                the number of samples and the second dimension is the
                dimension of the latent space.

        Returns:
            dict: State of the VAE.

        """
        encoder_state = self.encoder(x)
        # Forward the statistics to the latent model.
        p_np_params, acc_stats = self.latent_model.expected_natural_params(
                encoder_state['exp_T'])

        # Samples of the latent variable using the reparameterization
        # "trick". "z" is a L x N x K tensor where L is the number of
        # samples for the reparameterization "trick", N is the number
        # of frames and K is the dimension of the latent space.
        if self.sample:
            samples = []
            for i in range(self.nsamples):
                samples.append(self.encoder.sample(encoder_state))
            samples = torch.stack(samples)

            decoder_state = self.decoder(samples)
        else:
            decoder_state = self.decoder(encoder_state['means'])

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
        llh = self.decoder.log_likelihood(x, state['decoder_state']).sum()
        kl = self.encoder.kl_div(state['encoder_state'], state['p_np_params']).sum()
        kl *= kl_weight

        return -(llh - kl) / x.size(0), llh, kl


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
        log_base_measure = -.5 * math.log(2 * math.pi)
        llh = torch.sum((nparams * s_stats).sum(dim=1) + log_base_measure)

        # normalize by the number of samples (from the
        # reparameterization trick.)
        if len(nparams.size()) > 2:
            llh /= nparams.size(0)

        return llh

    @staticmethod
    def sample(state):
        '''Sample data using the reparametization trick.'''
        mean = state['mean']
        std_dev = torch.exp(.5 * state['std_dev'])
        noise = Variable(torch.randn(*mean.size()))
        return mean + std_dev * noise

    @staticmethod
    def kl_div(state, p_nparams):
        '''KL divergence between the posterior distribution and the
        prior.

        '''
        return ((state['natural_params'] - p_nparams) * state['exp_T']).sum(dim=1)

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
                (-.5 * diag_prec * mean ** 2).view(*x.size()),
                (.5 * torch.log(diag_prec)).view(*x.size())
            ], dim=-1)
        }


class MLPNormalIso(nn.Module):
    """Neural-Network ending with a double linear projection
    providing the mean and the logarithm of the isotropic covariance
    matrix.

    """

    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logvar = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if len(x.size()) > 2:
            x_reshaped = x.view(-1, x.size(2))
            logvar_size = (x.size(0), x.size(1), 1)
        else:
            x_reshaped = x
            logvar_size = (x.size(0), 1)

        h = self.structure(x_reshaped)

        return {
            'means': self.hid_to_mu(h).view(*x.size()),
            'logvars': self.hid_to_logvar(h).view(*logvar_size)
        }

    @staticmethod
    def log_likelihood(x, state):
        """Log-likelihood of the data x

        Args:
            state (dict): Current state of the neural network.

        Returns:
            torch.Variable: Per-frame log-likelihood.

        """
        mus = state['means']
        logvars = state['logvars']
        llh = -.5 * (math.log(2 * math.pi)
            + logvars + (x - mus).pow(2) * (-logvars).exp()).sum()

        # normalize by the number of samples (from the
        # reparameterization trick.)
        if len(mus.size()) > 2:
            llh /= mus.size(0)

        return llh

    @staticmethod
    def sample(state):
        """Sample data using the reparametization trick.

        Args:
            state (dict): Current state of the MLP.

        Returns:
            torch.Variable: Samples from the Normal MLP that can be
                differentiated w.r.t. the parameters of the MLP.

        """
        mus = state['means']
        std_devs = torch.exp(.5 * state['logvars'])
        noise = Variable(torch.randn(mus.size(0), mus.size(1)))
        return mus + std_devs * noise

    @staticmethod
    def expected_sufficient_statistics(state):
        """Expected value of the sufficient statistics (x, x**2).

        Args:
            state (dict): Current state of the MLP.

        Returns:
            torch.Variable: Expected value of x.
            torch.Variable: Expected value of x**2

        """
        return state['means'], torch.exp(state['logvars']) \
            + state['means'].pow(2)

    @staticmethod
    def kl_div(state, prior_state):
        """KL divergence between the posterior distribution and the
        prior.

        Args:
            state (dict): Current state of the MLP.
            latent_model_state (dict): Current state of the prior.

        """
        exp_x, exp_x2 = state['exp_x'], state['exp_x2']

        var = torch.exp(state['logvars'])
        q_np_linear, q_np_quadr = state['means'] / var, \
            -.5 * Variable(torch.ones(exp_x2.size(1))) / var
        p_np_linear, p_np_quadr = prior_state['np_linear'], \
            prior_state['np_quadr']
        exp_log_norm = prior_state['exp_log_norm']

        kl = ((q_np_linear - p_np_linear) * exp_x).sum(dim=1)
        kl += ((q_np_quadr - p_np_quadr) * exp_x2).sum(dim=1)
        kl += exp_log_norm - log_norm(q_np_linear, q_np_quadr)

        return kl.sum()

