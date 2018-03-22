
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

from .model import ConjugateExponentialModel
from ..expfamily import NormalGammaPrior
from ..expfamily import NormalWishartPrior
from ..expfamily import kl_div
from ..expfamily import _normalwishart_split_nparams



#######################################################################
# Normal model
#######################################################################

class Normal(ConjugateExponentialModel, metaclass=abc.ABCMeta):
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

    @property
    @abc.abstractmethod
    def count(self):
        'Number of data points used to estimate the parameters.'
        NotImplemented

    def kl_div_posterior_prior(self):
        '''KL divergence between the posterior and prior distribution.

        Returns:
            float: KL divergence.

        '''
        return kl_div(self.posterior, self.prior)

    def natural_grad_update(self, acc_stats, scale, lrate):
        '''Perform a natural gradient update of the posteriors'
        parameters.

        Args:
            acc_stats (dict): Accumulated statistics.
            scale (float): Scale of the sufficient statistics.
            lrate (float): Learning rate.

        '''
        # Compute the natural gradient.
        natural_grad = self.prior.natural_params + scale * acc_stats \
            - self.posterior.natural_params

        # Update the posterior distribution.
        self.posterior.natural_params = ta.Variable(
            self.posterior.natural_params + lrate * natural_grad,
            requires_grad=True)


class NormalDiagonalCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def sufficient_statistics(X):
        return torch.cat([X ** 2, X, torch.ones_like(X), torch.ones_like(X)],
                         dim=-1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([mean ** 2 + var, mean, torch.ones_like(mean),
                          torch.ones_like(mean)], dim=-1)

    def __init__(self, prior, posterior):
        self.prior = prior
        self.posterior = posterior

    @property
    def mean(self):
        np1, np2, _, _ = self.posterior.expected_sufficient_statistics.view(4, -1)
        return np2 / (-2 * np1)

    @property
    def cov(self):
        np1, np2, _, _ = self.posterior.expected_sufficient_statistics.view(4, -1)
        return torch.diag(1/(-2 * np1))

    @property
    def count(self):
        np1, _, _, np4 = self.posterior.expected_sufficient_statistics.view(4, -1)
        return float((self.posterior.natural_params[-1] + 1) /  (-4 * np1[-1]))

    def exp_llh(self, X, accumulate=False):
        T = self.sufficient_statistics(X)
        exp_natural_params = self.posterior.expected_sufficient_statistics

        # Note: the lognormalizer is already included in the expected
        # value of the natural parameters.
        exp_llh = T @ exp_natural_params - \
            .5 * X.shape[1] * math.log(2 * math.pi)

        if accumulate:
            acc_stats = T.sum(dim=0)
            return exp_llh, acc_stats

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
        self.prior = prior
        self.posterior = posterior

    @property
    def mean(self):
        nparams = self.posterior.expected_sufficient_statistics
        np1, np2, _, _, _ = _normalwishart_split_nparams(nparams)
        return torch.inverse(-2 * np1) @ np2

    @property
    def cov(self):
        nparams = self.posterior.expected_sufficient_statistics
        np1, _, _, _, _ = _normalwishart_split_nparams(nparams)
        return torch.inverse(-2 * np1)

    @property
    def count(self):
        return float(self.posterior.natural_params[-1])

    def exp_llh(self, X, accumulate=False):
        T = self.sufficient_statistics(X)
        exp_natural_params = self.posterior.expected_sufficient_statistics

        # Note: the lognormalizer is already included in the expected
        # value of the natural parameters.
        exp_llh = T @ exp_natural_params - \
            .5 * X.shape[1] * math.log(2 * math.pi)

        if accumulate:
            acc_stats = T.sum(dim=0)
            return exp_llh, acc_stats

        return exp_llh


#######################################################################
# NormalSet model
#######################################################################


class NormalSet(metaclass=abc.ABCMeta):
    'Set Normal density models.'

    def __init__(self, components):
        self.components = components

    def __len__(self):
        return len(self.components)

    def __getitem__(self, key):
        return self.components[key]

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics_from_mean_var(mean, var):
        NotImplemented

    def kl_div_posterior_prior(self):
        retval = 0
        for comp in self.components:
            retval += kl_div(comp.posterior, comp.prior)
        return retval

    def natural_grad_update(self, acc_stats, scale, lrate):
        for i, comp in enumerate(self.components):
            prior, post = comp.prior, comp.posterior
            natural_grad = prior.natural_params + scale * acc_stats[i] \
                - post.natural_params
            post.natural_params = ta.Variable(
                post.natural_params + lrate * natural_grad, requires_grad=True)


class NormalDiagonalCovarianceSet(NormalSet):
    'Set Normal density models with diagonal covariance.'

    def __init__(self, prior, posteriors):
        super().__init__([NormalDiagonalCovariance(prior, post)
                          for post in posteriors])


    def sufficient_statistics(X):
        return NormalDiagonalCovariance.sufficient_statistics(X)

    def sufficient_statistics_from_mean_var(mean, var):
        return NormalDiagonalCovariance.sufficient_statistics_from_mean_var(mean, var)

class NormalFullCovarianceSet(NormalSet):
    'Set Normal density models with full covariance.'

    def __init__(self, prior, posteriors):
        super().__init__([NormalFullCovariance(prior, post)
                          for post in posteriors])

    def sufficient_statistics(X):
        return NormalFullCovariance.sufficient_statistics(X)

    def sufficient_statistics_from_mean_var(mean, var):
        return NormalFullCovariance.sufficient_statistics_from_mean_var(mean, var)

