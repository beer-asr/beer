
"""Bayesian Normal distribution with prior over the mean and
covariance matrix.

"""

import abc
import math
from .model import ConjugateExponentialModel
from ..expfamily import NormalGammaPrior, NormalWishartPrior, kl_div
import torch
import torch.autograd as ta
import numpy as np
import math


class Normal(ConjugateExponentialModel, metaclass=abc.ABCMeta):
    '''Abstract Base Class for the Normal distribution model.'''


    @staticmethod
    @abc.abstractmethod
    def create(prior_mean, prior_cov=None, prior_count=1., random_init=False):
        '''Create a Normal distribution.

        Args:
            prior_mean (Tensor): Expected mean.
            prior_cov (Tensor): Expected covariance matrix.
            prior_count (float): Strength of the prior.
            random_init (boolean): If true, initialize the expected
                mean of the posterior randomly.

        Returns:
            ``Normal``: An initialized Normal distribution.

        '''
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        """Compute the sufficient statistics of the data.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        """Compute the sufficient statistics of the data.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics_from_mean_var(mean, var):
        """Compute the sufficient statistics of the data specified
        in term of a mean and variance for each data point.

        Args:
            mean (numpy.ndarray): Means for each point of the data.
            var (numpy.ndarray): Variances for each point (and
                dimension) of the data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
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

    @abc.abstractmethod
    def expected_natural_params(self, mean, var):
        '''Expected value of the natural parameters of the model.

        Args:
            mean (numpy.ndarray): Mean for each data point.
            var (numpy.ndarray): Variances for each point/dimension.

        Returns:
            (numpy.ndarray): Expected natural parameters.
            (numpy.ndarray): Accumulated statistics of the data.

        '''
        NotImplemented

    def kl_div_posterior_prior(self):
        """KL divergence between the posterior and prior distribution.

        Returns:
            float: KL divergence.

        """
        return kl_div(self.posterior, self.prior)

    def natural_grad_update(self, acc_stats, scale, lrate):
        """Perform a natural gradient update of the posteriors'
        parameters.

        Args:
            acc_stats (dict): Accumulated statistics.
            scale (float): Scale of the sufficient statistics.
            lrate (float): Learning rate.

        """
        # Compute the natural gradient.
        natural_grad = self.prior.natural_params + scale * acc_stats \
            - self.posterior.natural_params

        # Update the posterior distribution.
        self.posterior.natural_params = ta.Variable(
            self.posterior.natural_params + lrate * natural_grad,
            requires_grad=True)

    def split(self, prior_count=1.):
        '''Split the distribution into two Normal distribution (of the
        same type) by moving their mean by +- one standard deviation.

        Args:
            prior_count (float): Prior count to reset the distirbution.

        Returns:
            ``Normal``: First Normal distribution.
            ``Normal``: Second Normal distribution.

        '''
        dim = len(self.mean)
        evals, evecs = np.linalg.eigh(self.cov)
        basis = evals

        dist1 = self.create(dim=dim,
            mean=self.mean + evecs.T @  np.sqrt(basis),
            cov=self.cov, prior_count=prior_count)
        dist2 = self.create(dim=dim,
            mean=self.mean - evecs.T @  np.sqrt(basis),
            cov=self.cov, prior_count=prior_count)

        return dist1, dist2


class NormalDiagonalCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def create(prior_mean, prior_cov, prior_count=1., random_init=False):
        diag_cov = prior_prec if len(prior_cov.size()) == 1 else \
            torch.diag(prior_cov)
        diag_prec = 1. / diag_cov

        if prior_mean.size(0) != diag_prec.size(0):
            raise ValueError('Dimension mismatch: mean {} != cov {}'.format(
                prior_mean.size(0), diag_prec.size(0)))
        prior = NormalGammaPrior(prior_mean, diag_prec, prior_count)
        if random_init:
            rand_mean = np.random.multivariate_normal(prior_mean.numpy(),
                np.diag(diag_prec.numpy()))
            rand_mean = torch.from_numpy(rand_mean)
            rand_mean = rand_mean.type(prior_mean.type())
            posterior = NormalGammaPrior(rand_mean, diag_prec, prior_count)
        else:
            posterior = NormalGammaPrior(prior_mean, diag_prec, prior_count)

        return NormalDiagonalCovariance(prior, posterior)

    @staticmethod
    def sufficient_statistics(X):
        return torch.cat([X ** 2, X, torch.ones_like(X), torch.ones_like(X)],
                         dim=-1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return np.c_[mean**2 + var, mean, np.ones_like(mean),
                     np.ones_like(mean)]

    def __init__(self, prior, posterior):
        '''Initialize the Bayesian normal distribution.

        Args:
            prior (``beer.NormalGammaPrior``): Prior over the
                means and precisions.
            posterior (``beer.NormalGammaPrior``): Prior over the
                means and precisions.

        '''
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
        return (self.posterior.natural_params[-1] + 1) /  (-4 * np1[-1])

    def expected_natural_params(self, mean, var):
        # TODO: pytorch version.
        T = self.sufficient_statistics_from_mean_var(mean, var)
        np1, np2, np3, np4 = self.posterior.grad_lognorm().reshape(4, -1)
        identity = np.eye(var.shape[1])
        np1 = (np1[:, None] * identity[None, :, :]).reshape(-1)
        return np.c_[np1[None], np2[None], np3.sum(axis=-1)[None],
                     np4.sum(axis=-1)[None]], \
             T.sum(axis=0)

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
    def __extract_natural_params(natural_params):
        # TODO: this code is already in the the 'expfamily' module.
        # We need to retrieve the 4 natural parameters organized as
        # follows:
        #   [ np1_1, ..., np1_D^2, np2_1, ..., np2_D, np3, np4]
        #
        # The dimension D is found by solving the polynomial:
        #   D^2 + D - len(self.natural_params[:-2]) = 0
        D = int(.5 * (-1 + math.sqrt(1 + 4 * len(natural_params[:-2]))))
        np1, np2 = natural_params[:int(D**2)].view(D, D), \
             natural_params[int(D**2):-2]
        np3, np4 = natural_params[-2:]
        return np1, np2, np3, np4, D

    @staticmethod
    def create(prior_mean, prior_cov, prior_count=1., random_init=False):
        if prior_mean.size(0) != prior_cov.size(0):
            raise ValueError('Dimension mismatch: mean {} != cov {}'.format(
                prior_mean.size(0), prior_cov.size(0)))
        prior = NormalWishartPrior(prior_mean, prior_cov, prior_count)
        if random_init:
            rand_mean = np.random.multivariate_normal(prior_mean.numpy(),
                prior_cov.numpy().astype(np.float64))
            rand_mean = torch.from_numpy(rand_mean)
            rand_mean = rand_mean.type(prior_mean.type())
            posterior = NormalWishartPrior(rand_mean, prior_cov, prior_count)
        else:
            posterior = NormalWishartPrior(prior_mean, prior_cov, prior_count)
        return NormalFullCovariance(prior, posterior)

    @staticmethod
    def sufficient_statistics(X):
        return torch.cat([(X[:, :, None] * X[:, None, :]).view(len(X), -1),
            X, torch.ones(X.size(0), 1).type(X.type()),
            torch.ones(X.size(0), 1).type(X.type())], dim=-1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        idxs = np.identity(mean.shape[1]).reshape(-1) == 1
        XX = (mean[:, :, None] * mean[:, None, :]).reshape(mean.shape[0], -1)
        XX[:, idxs] += var
        return np.c_[XX, mean, np.ones(len(mean)), np.ones(len(mean))]

    def __init__(self, prior, posterior=None):
        self.prior = prior
        self.posterior = posterior

    @property
    def mean(self):
        nparams = self.posterior.expected_sufficient_statistics
        np1, np2, _, _, _ = self.__extract_natural_params(nparams)
        return torch.inverse(-2 * np1) @ np2

    @property
    def cov(self):
        nparams = self.posterior.expected_sufficient_statistics
        np1, _, _, _, _ = self.__extract_natural_params(nparams)
        return torch.inverse(-2 * np1)

    @property
    def count(self):
        return self.posterior.natural_params[-1]

    def expected_natural_params(self, mean, var):
        # TODO: pytorch version.
        T = self.sufficient_statistics_from_mean_var(mean, var)
        return self.posterior.grad_lognorm()[None, :], \
             T.sum(axis=0)

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

