
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


def _structure_output_dim(structure):
    'Find the output dimension of a given structure.'
    for transform in reversed(structure):
        if isinstance(transform, nn.Linear):
            s_out_dim = transform.out_features
            break
    return s_out_dim


class MLPModel(nn.Module, metaclass=abc.ABCMeta):
    '''Base class for the encoder / decoder neural network of
    the VAE. The output of this network are the parameters of a
    conjugate exponential model.

    '''

    def __init__(self, structure, outputs):
        '''Initialize the ``MLPModel``.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            outputs (list): List of output dimension for the
                parameters of the model.

        '''
        super().__init__()
        self.structure = structure
        s_out_dim = _structure_output_dim(structure)
        self.output_layer = nn.ModuleList()
        for i, outdim in enumerate(outputs):
            self.output_layer.append(nn.Linear(s_out_dim, outdim))

        # Make sure that by default the encoder/decoder has a small
        # variance.
        # TODO: should become a parameter.
        self.output_layer[-1].bias.data += -1

    def forward(self, X):
        h = self.structure(X)
        outputs = [transform(h) for transform in self.output_layer]
        return [outputs[0] + X, outputs[1]]


class MLPNormalDiag(MLPModel):
    '''Neural-Network ending with a double linear projection
    providing the mean and the logarithm of the diagonal of the
    covariance matrix.

    '''

    def __init__(self, structure, dim):
        '''Initialize a ``MLPNormalDiag`` object.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            dim (int): Desired dimension of the modeled random
                variable.

        '''
        super().__init__(structure, [dim, dim])

    def forward(self, X):
        mean, logvar = super().forward(X)
        return MLPStateNormalDiagonalCovariance(mean, torch.exp(logvar))


class MLPNormalIso(MLPModel):
    '''Neural-Network ending with a double linear projection
    providing the mean and the isotropic covariance matrix.

    '''

    def __init__(self, structure, dim):
        '''Initialize a ``MLPNormalDiag`` object.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            dim (int): Desired dimension of the modeled random
                variable.

        '''
        super().__init__(structure, [dim, 1])

    def forward(self, X):
        mean, logvar = super().forward(X)
        ones = Variable(torch.ones(mean.size(1)).type(X.type()))
        return MLPStateNormalDiagonalCovariance(mean, ones * torch.exp(logvar))


class MLPStateNormalDiagonalCovariance:

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


class MLPBernoulli(MLPModel):
    '''Neural-Network ending with a linear projection
    providing the mean of a Bernoulli distribution.

    '''

    def __init__(self, structure, dim):
        '''Initialize a ``MLPBernoulli`` object.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            dim (int): Desired dimension of the modeled random
                variable.

        '''
        super().__init__(structure, [dim])

    def forward(self, X):
        mu = super().forward(X)[0]
        return BernoulliState(F.sigmoid(mu))


class BernoulliState:
    ''' Bernoulli distribution, to be an output of a MLP.

    TODO -- as follows from the difference between the first line and
    the name of the class, something smells here. A lot.

    '''
    def __init__(self, mu):
        self.mu = mu

    def log_likelihood(self, T):
        dim0, dim1 = self._nparams.size()
        return torch.sum(T * self._nparams.view(1, dim0, dim1) , dim=-1) - \
            .5 * self.mean.size(0) * math.log(2 * math.pi)

    def log_likelihood(self, X):
        per_pixel_bce = X * self.mu.log() + (1.0 - X) * (1 - self.mu).log()
        return per_pixel_bce.sum(dim=-1)

