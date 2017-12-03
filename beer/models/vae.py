
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


    def __init__(self, encoder, decoder, latent_model, nb_samples):
        """Initialize the VAE.

        Args:
            encoder (``VAEComponent``): Encoder of the VAE.
            decoder (``VAEComponent``): Decoder of the VAE.
            latent_model(``Model``): Bayesian Model for the prior over
                the latent space.
            nb_samples (int): Number of samples to approximate the
                expectation of the log-likelihood.

        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nb_samples = nb_samples

    def reparameterize(self, mu, logvar):
        """Re-parameterization "trick" to allow the derivation of the
        objective function w.r.t. to the encoder's parameters.

        Args:
            mu (torch.Variable): Per-frame mean of the Normal
                distribution over the latent variable.
            logvar (torch.Variable): Per-framce logarithm of the
                variance of the Normal distribution over the latent
                variable.

        Note:
            If the model is not in training mode i.e.:

                 >>> model.training
                 False

            this function is effectless.

        Returns:
            torch.Variable: Samples from the posetrior distributions.

        """
        if self.training:
            z_samples = []
            for i in range(self.nb_samples):
                std = logvar.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                z_samples.append(eps.mul(std).add_(mu))

            return torch.stack(z_samples)
        else:
            return mu.unsqueeze(0)

    def forward(self, x):
        """Forward data through the VAE model.

        Args:
            x (torch.Variable): Data to process. The first dimension is
                the number of samples and the second dimension is the
                dimension of the latent space.

        Returns:
            torch.Variable: Per-frame and per-sample mean of the
                likelihood function.
            torch.Variable: Per-frame and per-sample logarithm of the
                variable of the likelihood function.
            torch.Variable: Per-frame mean of the posterior
                distribution.
            torch.Variable: Per-frame logarithm of the variance of the
                posterior distribution.

        """
        # Mean and (log-)variance of the posterior distribution over
        # the latent space.
        mu, logvar = self.encoder(x)

        # Samples of the latent variable using the reparameterization
        # "trick". "z" is a L x N x K tensor where L is the number of
        # samples for the reparameterization "trick", N is the number
        # of frames and K is the dimension of the latent space.
        z = self.reparameterize(mu, logvar)

        # Forward the samples through the decoder.
        obs_mu, obs_logvar = self.decoder(z.view(-1, 2))

        # Re-organize the parameters of the likelihood as a L x N x K
        # tensor
        obs_mu = obs_mu.view(-1, x.size(0), 2)
        obs_logvar = obs_logvar.view(-1, x.size(0), 2)

        return obs_mu, obs_logvar, mu, logvar

    def loss(self, x, obs_mu, obs_logvar, z_mu, z_logvar):
        """Loss function of the VAE. This is the negative of the
        variational objective function i.e.:

            loss = - ( E_q [ ln p(X|Z) ] - KL( q(z) || p(z) ) )

        Args:
            x (torch.Variable): Data on which to estimate the loss.
            obs_mu (torch.Variable): Mean of the likelihood function.
            obs_logvar (torch.Variable): (Log-)variance of the
                likelihood function.
            z_mu (torch.Variable): Mean of the posterior distributions.
            z_logvar (torch.Variable): (Log-)variance of the posterior
                distributions.

        Returns:
            torch.Variable: Symbolic computation of the loss function.

        """
        LLH = self.gauss_LLH(x, obs_mu, obs_logvar)

        # Natural parameters and expected sufficient statistics of the
        # posteriors.
        q_np_linear, q_np_quadr, stats_linear, stats_quadr = \
            naturals_from_mu_logvar(z_mu, z_logvar)

        # Accumulate statistics.
        acc_stats = self.latent_model.accumulate_stats(
            stats_linear.data.numpy().T, stats_quadr.data.numpy().T)

        # Expected value of the natural parameters of the Gaussian and
        # the log-normalizer.
        p_np_linear, p_np_quadr, exp_log_norm = \
            self.latent_model.expected_natural_params(
                stats_linear.data.numpy().T, stats_quadr.data.numpy().T)

        # Make sure they are torch Variable.
        p_np_linear = to_torch_variable(p_np_linear)
        p_np_quadr = to_torch_variable(p_np_quadr)
        exp_log_norm = to_torch_variable(exp_log_norm)

        KLD = kld(q_np_linear, q_np_quadr, stats_linear, stats_quadr,
                  p_np_linear, p_np_quadr, exp_log_norm)

        return -(LLH - KLD), LLH, KLD, acc_stats

    def gauss_LLH(self, x, obs_mu, obs_logvar):
        LLH = 0.5 * (
            (-obs_logvar) +
            (- (x-obs_mu).pow(2) * (-obs_logvar).exp())
        ).sum()
        return LLH / obs_mu.size(0)


class GaussianMLP(nn.Module):
    """ Neural-Network ending with a double linear projectiong
    providing the mean and the (log-)variance a of a Normal
    distribution given the input.

    """

    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logvar = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        h = self.structure(x)
        return self.hid_to_mu(h), self.hid_to_logvar(h)


class IsotropicGaussian:
    def __init__(self, dim, log_var=0.0):
        self._dim = dim
        self._log_var = Variable(torch.FloatTensor([log_var]))

    def kld(self, mu, logvar):
        assert(mu.size(1) == self._dim)

        KLD = 0.5 * torch.sum(
            (self._log_var - 1) +
            (- logvar) +
            logvar.exp()/self._log_var.exp() +
            mu.pow(2)/self._log_var.exp()
        )
        return KLD

    def update(self, mu, logvar):
        pass


class NaturalIsotropicGaussian:
    def __init__(self, dim, log_var=0.0):
        var = math.exp(log_var)
        self._np_linear = Variable(torch.FloatTensor([0]*dim))
        self._np_quadr = Variable(torch.FloatTensor([-0.5*var]*dim))

    def accumulate_stats(self, X1, X2):
        return None

    def expected_natural_params(self, stats_linear, stats_quadr):
        # stats_linear and stats_quadr are ignored as the parameters
        # of the Gaussian are fixed.
        np_linear, np_quadr = self._np_linear.view(1, -1), \
            self._np_quadr.view(1, -1)
        exp_log_norm = log_norm(np_linear, np_quadr)
        return np_linear, np_quadr, exp_log_norm

    def natural_grad_update(self, acc_stats, scale, lrate):
        pass

def to_torch_variable(variable):
    """If the given variable is a numpy.ndarray convert it to a torch
    Variable convert it.

    """
    if type(variable) == np.ndarray:
        return Variable(torch.FloatTensor(variable))
    return variable

def kld(q_np_linear, q_np_quadr, stats_linear, stats_quadr, p_np_linear,
        p_np_quadr, exp_log_norm):
    KLD = ((q_np_linear - p_np_linear) * stats_linear).sum(dim=1)
    KLD += ((q_np_quadr - p_np_quadr) * stats_quadr).sum(dim=1)
    KLD += exp_log_norm
    KLD += -log_norm(q_np_linear, q_np_quadr)
    KLD = KLD.sum()

    return KLD

def log_norm(np_linear, np_quadr):
    return - 0.5 * torch.log((-2*np_quadr)).sum(dim=1) - 0.25 * (np_linear.pow(2) * (1.0/np_quadr)).sum(dim=1)

def naturals_from_mu_logvar(mu, logvar):
    var = logvar.exp()
    stats_linear = mu
    stats_quadr = mu.pow(2) + var

    np_linear = mu / var
    np_quadr = -0.5 / var
    return np_linear, np_quadr, stats_linear, stats_quadr


class MLPNormalDiag(nn.Module):
    """ Neural-Network ending with a double linear projectiong
    providing the mean and the logarithm of the diagonal of the
    covariance matrix.

    """

    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logvar = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        h = self.structure(x)
        return {
            'means': self.hid_to_mu(h),
            'logvars': self.hid_to_logvar(h)
        }

    @staticmethod
    def log_likelihood(x, state):
        """Log-likelihood of the data x

        Args:
            state (dict): Current state of the neural network.

        Returns:
            torch.Variable: Per-frame log-likelihood.

        """
        mu = state['means']
        logvar = state['logvars']
        return -.5 * (mu.size(1) * math.log(2 * math.pi) + logvar +
            (x - mu).pow(2) * (-logvar).exp()
        ).sum(1)

