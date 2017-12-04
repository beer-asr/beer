
"""Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

"""

import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class VAE(nn.Module):
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

        # Expected value of the sufficient statistics.
        exp_x, exp_x2 = self.encoder.expected_sufficient_statistics(
            encoder_state)
        encoder_state = {**encoder_state, 'exp_x': exp_x, 'exp_x2': exp_x2}

        # Forward the statistics to the latent model.
        latent_model_state = self.latent_model(exp_x, exp_x2)

        # Samples of the latent variable using the reparameterization
        # "trick". "z" is a L x N x K tensor where L is the number of
        # samples for the reparameterization "trick", N is the number
        # of frames and K is the dimension of the latent space.
        samples = []
        for i in range(self.nsamples):
            samples.append(self.encoder.sample(encoder_state))
        samples = torch.stack(samples)

        decoder_state = self.decoder(samples)

        return {
            'encoder_state': encoder_state,
            'latent_model_state': latent_model_state,
            'decoder_state': decoder_state
        }

    def loss(self, x, state):
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
        kl = self.encoder.kl_div(state['encoder_state'],
                                 state['latent_model_state'])

        return -(llh - kl) / x.size(0), llh, kl


class NaturalIsotropicGaussian:
    def __init__(self, dim, log_var=0.0):
        var = math.exp(log_var)
        self._np_linear = Variable(torch.FloatTensor([0]*dim))
        self._np_quadr = Variable(torch.FloatTensor([-0.5 / var]*dim))

    def __call__(self, exp_x, exp_x2):
        np_linear = Variable(torch.ones(exp_x.size())) * self._np_linear
        np_quadr = Variable(torch.ones(exp_x2.size())) * self._np_quadr
        exp_log_norm = log_norm(np_linear, np_quadr)
        return {
            'np_linear': np_linear,
            'np_quadr': np_quadr,
            'exp_log_norm': exp_log_norm
        }

    def expected_natural_params(self, stats_linear, stats_quadr):
        # stats_linear and stats_quadr are ignored as the parameters
        # of the Gaussian are fixed.
        np_linear, np_quadr = self._np_linear.view(1, -1), \
            self._np_quadr.view(1, -1)
        exp_log_norm = log_norm(np_linear, np_quadr)
        return np_linear, np_quadr, exp_log_norm

    def natural_grad_update(self, acc_stats, scale, lrate):
        pass

def log_norm(np_linear, np_quadr):
    return - 0.5 * torch.log((-2*np_quadr)).sum(dim=1) - 0.25 * (np_linear.pow(2) * (1.0/np_quadr)).sum(dim=1)


class MLPNormalDiag(nn.Module):
    """Neural-Network ending with a double linear projection
    providing the mean and the logarithm of the diagonal of the
    covariance matrix.

    """

    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logvar = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        if len(x.size()) > 2:
            x_reshaped = x.view(-1, x.size(2))
        else:
            x_reshaped = x

        h = self.structure(x_reshaped)
        return {
            'means': self.hid_to_mu(h).view(*x.size()),
            'logvars': self.hid_to_logvar(h).view(*x.size())
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
        llh = -.5 * (math.log(2 * math.pi) + logvars +
            (x - mus).pow(2) * (-logvars).exp()
        ).sum()

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
        inv_std_devs = torch.exp(-.5 * state['logvars'])
        noise = Variable(torch.randn(mus.size(0), mus.size(1)))
        return mus + inv_std_devs * noise

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
        q_np_linear, q_np_quadr = state['means'] / var, -.5 / var
        p_np_linear, p_np_quadr = prior_state['np_linear'], \
            prior_state['np_quadr']
        exp_log_norm = prior_state['exp_log_norm']

        kl = ((q_np_linear - p_np_linear) * exp_x).sum(dim=1)
        kl += ((q_np_quadr - p_np_quadr) * exp_x2).sum(dim=1)
        kl += exp_log_norm - log_norm(q_np_linear, q_np_quadr)

        return kl.sum()


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
        inv_std_devs = torch.exp(-.5 * state['logvars'])
        noise = Variable(torch.randn(mus.size(0), mus.size(1)))
        return mus + inv_std_devs * noise

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

