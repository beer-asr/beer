
'''Implementation of various Multi-Layer Perceptron with specific
final layer corresponding to the parameters of a distribution.
This kind of MLP are used with model combining deep neural network
and bayesian model (Variational Auto-Encoder and its variant).

'''

import abc
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from .normal import NormalDiagonalCovariance
from .normal import normal_diag_natural_params


class MLPModel(nn.Module):
    '''Base class for the encoder / decoder neural network of
    the VAE. The output of this network are the parameters of a
    conjugate exponential model.

    '''

    def __init__(self, structure, s_out_dim, dist_builder):
        '''Initialize the ``MLPModel``.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            outputs (list): List of output dimension for the
                parameters of the model.

        '''
        super().__init__()
        self.structure = structure
        self._dist_builder = dist_builder

        self.output_layer = nn.ModuleList()
        for outdim in self._dist_builder.required_params_shape():
            self.output_layer.append(nn.Linear(s_out_dim, outdim))

    def forward(self, X):
        h = self.structure(X)
        dist_params = [transform(h) for transform in self.output_layer]
        return self._dist_builder(dist_params)


class NormalDiagBuilder:
    def __init__(self, dim):
        self.dim = dim

    def required_params_shape(self):
        return [self.dim, self.dim]

    def __call__(self, params):
        mean = params[0]
        logvar = params[1]
        return NormalDiagonalCovariance_MLP(mean, logvar.exp())


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


class BernoulliBuilder:
    def __init__(self, dim):
        self.dim = dim

    def required_params_shape(self):
        return [self.dim]

    def __call__(self, params):
        mu = params[0]
        return Bernoulli_MLP(F.sigmoid(mu))


class Bernoulli_MLP:
    ''' Bernoulli distribution, to be an output of a MLP.

    '''
    def __init__(self, mu):
        self.mu = mu

    def log_likelihood(self, X):
        per_pixel_bce = X * self.mu.log() + (1.0 - X) * (1 - self.mu).log()
        return per_pixel_bce.sum(dim=-1)

