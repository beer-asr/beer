
'''Implementation of various Multi-Layer Perceptron with specific
final layer corresponding to the parameters of a distribution.
This kind of MLP are used with model combining deep neural network
and bayesian model (Variational Auto-Encoder and its variant).

'''

import abc
import math

import torch
from torch import nn
from torch.autograd import Variable


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

        # Get the ouput dimension of the structure.
        for transform in reversed(structure):
            if isinstance(transform, nn.Linear):
                s_out_dim = transform.out_features
                break

        # Create the specific output layer.
        self.output_layer = nn.ModuleList()
        for i, outdim in enumerate(outputs):
            self.output_layer.append(nn.Linear(s_out_dim, outdim))

    def forward(self, X):
        h = self.structure(X)
        outputs = [transform(h) for transform in self.output_layer]
        return outputs


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
        mean, logprec = super().forward(X)
        return MLPStateNormal(mean, torch.exp(logprec))


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
        mean, logprec = super().forward(X)
        return MLPStateNormal(mean,
            torch.exp(logprec) * Variable(torch.ones(mean.size(1)).float()))


class MLPEncoderState(metaclass=abc.ABCMeta):
    '''A state of a ``MLPModel`` is the wrapper around a given values
    of the parameters of the model. The ``MLPModel`` defines the
    structure of the MLP whereas the ``MLPModelState`` is the set
    of distributions corresponding to the data forwarded through
    a ``MLPModel``.

    '''

    @property
    def mean(self):
        'Mean of each distribution.'
        return self._mean

    @property
    def prec(self):
        'Diagonal of the precision matrix for each distribution.'
        return self._prec

    @abc.abstractmethod
    def sample(self):
        'sample data using the reparametization trick.'
        NotImplemented

    @abc.abstractmethod
    def kl_div(self, p_nparams):
        'kl divergence between the posterior and prior distribution.'
        NotImplemented


class MLPDecoderState(metaclass=abc.ABCMeta):
    'Abstract Base Class for the state of a the MLPDecoder.'

    @abc.abstractmethod
    def natural_params(self):
        'Natural parameters for each distribution.'
        NotImplemented

    @abc.abstractmethod
    def log_base_measure(self, X):
        'Natural parameters for each distribution.'
        NotImplemented

    @abc.abstractmethod
    def sufficient_statistics(self, X):
        'Sufficient statistics of the given data.'
        NotImplemented

    def log_likelihood(self, X, nsamples=1):
        'Log-likelihood of the data.'
        s_stats = self.sufficient_statistics(X)
        nparams = self.natural_params()
        log_bmeasure = self.log_base_measure(X)
        nparams = nparams.view(nsamples, X.size(0), -1)
        return torch.sum(nparams * s_stats, dim=-1) + log_bmeasure


class MLPStateNormal(MLPEncoderState, MLPDecoderState):

    def __init__(self, mean, prec):
        self._mean = mean
        self._prec = prec

    def exp_T(self):
        idxs = torch.arange(0, self.mean.size(1)).long()
        XX = self.mean[:, :, None] * self.mean[:, None, :]
        XX[:, idxs, idxs] += 1 / self.prec
        return torch.cat([XX.view(self.mean.size(0), -1), self.mean,
                          Variable(torch.ones(self.mean.size(0), 2))], dim=-1)

    @property
    def std_dev(self):
        return 1 / torch.sqrt(self.prec)

    def natural_params(self):
        identity = Variable(torch.eye(self.mean.size(1)))
        np1 = -.5 * self.prec[:, None] * identity[None, :, :]
        np1 = np1.view(self.mean.size(0), -1)
        np2 = self.prec * self.mean
        np3 = -.5 * (self.prec * (self.mean ** 2)).sum(-1)[:, None]
        np4 = .5 * torch.log(self.prec).sum(-1)[:, None]
        return torch.cat([np1, np2, np3, np4], dim=-1)

    def sample(self):
        noise = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std_dev * noise

    def kl_div(self, p_nparams):
        return ((self.natural_params() - p_nparams) * self.exp_T()).sum(dim=-1)

    def sufficient_statistics(self, X):
        XX = X[:, :, None] * X[:, None, :]
        return torch.cat([XX.view(X.size(0), -1), X,
                          Variable(torch.ones(X.size(0), 2).float())], dim=-1)

    def log_base_measure(self, X):
        return -.5 * X.size(-1) * math.log(2 * math.pi)

