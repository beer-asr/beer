
'''Bayesian Normal distribution with prior over the mean and
covariance matrix.


 Normal model
 ------------
   The ``Normal`` model is very simple model that fits a data with
   a Normal density. Practically, the Normal class is just an
   interface. It has 2 concrete implementations, one with diagonal
   covariance matrix and the other one with full covariance matrix.


NormalSet
--------
   The ``NormalSet`` object is not a model but rather a component of
   more comple model (e.g. GMM). It allows to have a set of Normal
   densities to have a shared prior distribution.


'''

import abc
from collections import namedtuple
import math

import torch
import torch.autograd as ta

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .bayesmodel import BayesianModel
from ..expfamilyprior import NormalGammaPrior
from ..expfamilyprior import NormalWishartPrior
from ..expfamilyprior import kl_div
from ..expfamilyprior import _normalwishart_split_nparams


#######################################################################
# Normal model
#######################################################################

class Normal(BayesianModel, metaclass=abc.ABCMeta):
    'Abstract Base Class for the Normal distribution model.'

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        '''Compute the sufficient statistics of the data.

        Args:
            X (Tensor): Data.

        Returns:
            (Tensor): Sufficient statistics of the data.

        '''
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics_from_mean_var(mean, var):
        '''Compute the sufficient statistics of the data specified
        in term of a mean and variance for each data point.

        Args:
            mean (Tensor): Means for each point of the data.
            var (Tensor): Variances for each point (and
                dimension) of the data.

        Returns:
            (Tensor): Sufficient statistics of the data.

        '''
        NotImplemented

    @property
    @abc.abstractmethod
    def mean(self):
        'Expected value of the mean w.r.t. posterior distribution.'
        NotImplemented

    @property
    @abc.abstractmethod
    def cov(self):
        '''Expected value of the covariance matrix w.r.t posterior
         distribution.

        '''
        NotImplemented

    def accumulate(self, T, parent_message=None):
        return T.sum(dim=0)


class NormalDiagonalCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def sufficient_statistics(X):
        return torch.cat([X ** 2, X, torch.ones_like(X), torch.ones_like(X)],
                         dim=-1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([(mean ** 2) + var, mean, torch.ones_like(mean),
                          torch.ones_like(mean)], dim=-1)

    def __init__(self, prior, posterior):
        super().__init__()
        self._mean_prec = BayesianParameter(prior, posterior)

    @property
    def mean(self):
        evalue = self._mean_prec.expected_value
        np1, np2, _, _ = evalue.view(4, -1)
        return np2 / (-2 * np1)

    @property
    def cov(self):
        evalue = self._mean_prec.expected_value
        np1, np2, _, _ = evalue.view(4, -1)
        return torch.diag(1/(-2 * np1))

    def forward(self, T, labels=None):
        exp_llh = T @ self._mean_prec.expected_value
        exp_llh -= .125 * T.size(1) * math.log(2 * math.pi)
        return exp_llh


class NormalFullCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def sufficient_statistics(X):
        return torch.cat([(X[:, :, None] * X[:, None, :]).view(len(X), -1),
            X, torch.ones(X.size(0), 1).type(X.type()),
            torch.ones(X.size(0), 1).type(X.type())], dim=-1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        idxs = torch.eye(mean.size(1)).view(-1) == 1
        XX = (mean[:, :, None] * mean[:, None, :]).view(mean.shape[0], -1)
        XX[:, idxs] += var
        return torch.cat([XX, mean, torch.ones(len(mean), 1).type(mean.type()),
            torch.ones(len(mean), 1).type(mean.type())], dim=-1)

    def __init__(self, prior, posterior):
        super().__init__()
        self._mean_prec = BayesianParameter(prior, posterior)

    @property
    def mean(self):
        evalue = self._mean_prec.expected_value
        np1, np2, _, _, _ = _normalwishart_split_nparams(evalue)
        return torch.inverse(-2 * np1) @ np2

    @property
    def cov(self):
        evalue = self._mean_prec.expected_value
        np1, _, _, _, _ = _normalwishart_split_nparams(evalue)
        return torch.inverse(-2 * np1)

    def forward(self, T, labels=None):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - T.size(1))))
        exp_natural_params = self._mean_prec.expected_value
        exp_llh = T @ self._mean_prec.expected_value
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh


#######################################################################
# NormalSet model
#######################################################################


class NormalSet(BayesianModel, metaclass=abc.ABCMeta):
    'Set Normal density models.'

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics_from_mean_var(mean, var):
        NotImplemented

    def __init__(self, components):
        super().__init__()
        self.components = components
        self.__parameters = BayesianParameterSet([
            BayesianParameter(comp.parameters[0].prior,
                              comp.parameters[0].posterior)
            for comp in self.components
        ])

    def _expected_nparams_as_matrix(self):
        return torch.cat([param.expected_value[None]
            for param in self.__parameters], dim=0)

    def forward(self, T, labels=None):
        return T @ self._expected_nparams_as_matrix().t()

    def accumulate(self, T, weights):
        return list(weights.t() @ T)


class NormalDiagonalCovarianceSet(NormalSet):
    'Set Normal density models with diagonal covariance.'

    def __init__(self, prior, posteriors):
        components = [
            NormalDiagonalCovariance(prior, post) for post in posteriors
        ]
        super().__init__(components)

    @staticmethod
    def sufficient_statistics(X):
        return NormalDiagonalCovariance.sufficient_statistics(X)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalDiagonalCovariance.sufficient_statistics_from_mean_var(mean,
            var)


class NormalFullCovarianceSet(NormalSet):
    'Set Normal density models with full covariance.'

    def __init__(self, prior, posteriors):
        components = [
            NormalFullCovariance(prior, post) for post in posteriors
        ]
        super().__init__(components)

    @staticmethod
    def sufficient_statistics(X):
        return NormalFullCovariance.sufficient_statistics(X)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalFullCovariance.sufficient_statistics_from_mean_var(mean,
            var)

