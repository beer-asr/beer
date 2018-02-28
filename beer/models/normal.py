
"""Bayesian Normal distribution with prior over the mean and
covariance matrix.

"""

import abc
from .model import ConjugateExponentialModel
from ..expfamily import NormalGammaPrior, NormalWishartPrior
import copy
import torch
import torch.autograd as ta
import numpy as np


class Normal(ConjugateExponentialModel, metaclass=abc.ABCMeta):
    '''Abstract Base Class for the Normal distribution model.'''


    @staticmethod
    @abc.abstractmethod
    def create(prior_mean, prior_prec=None, prior_count=1., random_init=False):
        '''Create a Normal distribution.

        Args:
            prior_mean (Tensor): Expected mean of the prior.
            prior_prec (Tensor): Expected precision matrix of the
                prior.
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
    def lognorm(self):
        """Expected value of the log-normalizer.

        Returns:
            float: expected value of the log-normalizer.

        """
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
        return self.posterior.kl_div(self.prior)

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
        self.posterior.natural_params += lrate * natural_grad

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
    """Bayesian Normal distribution with diagonal covariance matrix."""

    @staticmethod
    def create(prior_mean, prior_prec, prior_count=1., random_init=False):
        diag_prec = prior_prec if len(prior_prec.size()) == 1 else \
            torch.diag(prior_prec)

        if prior_mean.size() != diag_prec.size():
            raise ValueError('Dimension mismatch: mean {} != precision {}'.format(
                prior_mean.size(), diag_prec.size()))
        prior = NormalGammaPrior(prior_mean, diag_prec, prior_count)
        if random_init:
            rand_mean = np.random.multivariate_normal(vmean.data.numpy(),
                np.diag(vprec.data.numpy()))
            rand_vmean = torch.from_numpy(rand_mean),
            rand_vmean = rand_vmean.type(prior_mean.type())

            posterior = NormalGammaPrior(rand_vmean, vprec, prior_count)
        else:
            posterior = NormalGammaPrior(prior_mean, diag_prec, prior_count)

        return NormalDiagonalCovariance(prior, posterior)

    @staticmethod
    def sufficient_statistics(X):
        return np.c_[X**2, X, np.ones_like(X), np.ones_like(X)]

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
        return self.posterior.natural_params[-1]

    def lognorm(self):
        _, np2, np3, np4 = self.posterior.grad_lognorm().reshape(4, -1)
        return -np.sum(np3 + np4) - .5 * len(np2) * np.log(2 * np.pi)

    def expected_natural_params(self, mean, var):
        T = self.sufficient_statistics_from_mean_var(mean, var)
        np1, np2, np3, np4 = self.posterior.grad_lognorm().reshape(4, -1)
        identity = np.eye(var.shape[1])
        np1 = (np1[:, None] * identity[None, :, :]).reshape(-1)
        return np.c_[np1[None], np2[None], np3.sum(axis=-1)[None],
                     np4.sum(axis=-1)[None]], \
             T.sum(axis=0)

    def exp_llh(self, X, accumulate=False):
        T = self.sufficient_statistics(X)
        exp_natural_params = self.posterior.grad_lognorm()

        # Note: the lognormalizer is already included in the expected
        # value of the natural parameters.
        exp_llh = T @ exp_natural_params - .5 * X.shape[1] * np.log(2 * np.pi)

        if accumulate:
            acc_stats = T.sum(axis=0)
            return exp_llh, acc_stats

        return exp_llh


class NormalFullCovariance(Normal):
    """Bayesian Normal distribution with diagonal covariance matrix."""

    @staticmethod
    def create(mean, cov, prior_count=1., random_init=False):
        prior = NormalWishartPrior.from_std_parameters(
            mean,
            prior_count,
            cov,
            dim - 1 + prior_count
        )

        if random_init:
            posterior = NormalWishartPrior.from_std_parameters(
                np.random.multivariate_normal(mean, cov),
                prior_count,
                cov,
                dim - 1 + prior_count
            )
        else:
            posterior = None

        return NormalFullCovariance(prior, posterior)

    @staticmethod
    def sufficient_statistics(X):
        """Compute the sufficient statistics of the data.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
        return np.c_[(X[:, :, None] * X[:, None, :]).reshape(len(X), -1),
            X, np.ones(len(X)), np.ones(len(X))]

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        """Compute the sufficient statistics of the data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
        idxs = np.identity(mean.shape[1]).reshape(-1) == 1
        XX = (mean[:, :, None] * mean[:, None, :]).reshape(mean.shape[0], -1)
        XX[:, idxs] += var
        return np.c_[XX, mean, np.ones(len(mean)), np.ones(len(mean))]

    def __init__(self, prior, posterior=None):
        """Initialize the Bayesian normal distribution.

        Args:
            prior (``beer.priors.NormalWishartPrior``): Prior over the
                means and precisions.

        """
        self.prior = prior
        if posterior is not None:
            self.posterior = posterior
        else:
            self.posterior = copy.deepcopy(prior)

    @property
    def mean(self):
        grad = self.posterior.grad_lognorm()
        np1, np2, _, _, _ = NormalWishartPrior.extract_natural_params(grad)
        return np.linalg.inv(-2 * np1) @ np2

    @property
    def cov(self):
        grad = self.posterior.grad_lognorm()
        np1, _, _, _, _ = NormalWishartPrior.extract_natural_params(grad)
        return np.linalg.inv(-2 * np1)

    @property
    def count(self):
        return self.posterior.natural_params[-1]

    def lognorm(self):
        grad = self.posterior.grad_lognorm()
        _, _, np3, np4, D = NormalWishartPrior.extract_natural_params(grad)
        return -np.sum(np3 + np4 - .5 * D * np.log(2 * np.pi))

    def expected_natural_params(self, mean, var):
        T = self.sufficient_statistics_from_mean_var(mean, var)
        return self.posterior.grad_lognorm()[None, :], \
             T.sum(axis=0)

    def exp_llh(self, X, accumulate=False):
        T = self.sufficient_statistics(X)
        exp_natural_params = self.posterior.grad_lognorm()

        # Note: the lognormalizer is already included in the expected
        # value of the natural parameters.
        exp_llh = T @ exp_natural_params

        if accumulate:
            acc_stats = T.sum(axis=0)
            return exp_llh, acc_stats

        return exp_llh

