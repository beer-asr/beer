
'''Implementation of various Multi-Layer Perceptron with specific
final layer corresponding to the parameters of a distribution.
This kind of MLP are used with model combining deep neural network
and bayesian model (Variational Auto-Encoder and its variant).

'''

import torch
from torch.autograd import Variable

from .normal import NormalDiagonalCovariance
from .normal import normal_diag_natural_params


class NormalDiagonalCovariance_MLP:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self._nparams = normal_diag_natural_params(self.mean, self.var)

    def entropy(self):
        'Compute the per-frame entropy of the posterior distribution.'
        nparams = normal_diag_natural_params(self.mean, self.var)
        exp_T = NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            self.mean, self.var)
        return - (self._nparams * exp_T).sum(dim=-1)

    def kl_div(self, nparams_other):
        nparams = normal_diag_natural_params(self.mean, self.var)
        exp_T = NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            self.mean, self.var)
        return ((nparams - nparams_other) * exp_T).sum(dim=-1)

    def sample(self):
        noise = Variable(torch.randn(*self.mean.size()))
        return self.mean + noise * torch.sqrt(self.var)

    def log_likelihood(self, X):
        distance_term = 0.5 * (X - self.mean).pow(2) / self.var
        precision_term = 0.5 * self.var.log()
        return (-distance_term - precision_term).sum(dim=-1).mean(dim=0)


class Bernoulli_MLP:
    ''' Bernoulli distribution, to be an output of a MLP.

    '''
    def __init__(self, mu):
        self.mu = mu

    def log_likelihood(self, X):
        per_pixel_bce = X * self.mu.log() + (1.0 - X) * (1 - self.mu).log()
        return per_pixel_bce.sum(dim=-1)

