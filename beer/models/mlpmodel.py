
''' Implementation of distributions to be used by the
    VAE's encoder and decoder. User defined transformations
    (neural networks) are expected to produce them.
'''

import torch
from torch.autograd import Variable

from .normal import NormalDiagonalCovariance
from .normal import normal_diag_natural_params


class NormalDiagonalCovarianceMLP:
    ''' Normal distribution with diagonal covariance, to be an output of a MLP.
        It can be used both in observed and latent space, when a (conditionally)
        normal distribution is used in the latent space.

    '''

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self._nparams = normal_diag_natural_params(self.mean, self.var)

    def entropy(self):
        'Compute the per-frame entropy of the posterior distribution.'
        exp_T = NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            self.mean, self.var)
        return - (self._nparams * exp_T).sum(dim=-1)

    def kl_div(self, nparams_other):
        nparams = normal_diag_natural_params(self.mean, self.var)
        exp_s_stats = \
            NormalDiagonalCovariance.sufficient_statistics_from_mean_var(\
                self.mean, self.var)
        return ((nparams - nparams_other) * exp_s_stats).sum(dim=-1)

    def sample(self):
        noise = Variable(torch.randn(*self.mean.size()))
        return self.mean + noise * torch.sqrt(self.var)

    def log_likelihood(self, data):
        distance_term = 0.5 * (data - self.mean).pow(2) / self.var
        precision_term = 0.5 * self.var.log()
        return (-distance_term - precision_term).sum(dim=-1).mean(dim=0)


class BernoulliMLP:
    ''' Bernoulli distribution, to be an output of a MLP.
        Can only be used for modeling in the observed space, as it does not support
        sampling of KLD computation.

    '''
    def __init__(self, mu):
        self.mu = mu

    def log_likelihood(self, X):
        per_pixel_bce = X * self.mu.log() + (1.0 - X) * (1 - self.mu).log()
        return per_pixel_bce.sum(dim=-1)
