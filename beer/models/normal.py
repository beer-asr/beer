
"""Bayesian Normal distribution with prior over the mean and
covariance matrix.

"""

import abc
from .model import ConjugateExponentialModel
from ..priors import NormalGammaPrior
from ..priors import NormalWishartPrior
import copy
import numpy as np


class NormalDiagonalCovariance(ConjugateExponentialModel):
    """Bayesian Normal distribution with diagonal covariance matrix."""

    @staticmethod
    def create(dim, mean=None, cov=None, prior_count=1., random_init=False):
        if mean is None: mean = np.zeros(dim)
        if cov is None: cov = np.identity(dim)

        variances = np.diag(cov)
        prior = NormalGammaPrior.from_std_parameters(
            mean,
            np.ones(dim) * prior_count,
            ((1/variances) * prior_count),
            np.ones(dim) * prior_count
        )

        if random_init:
            posterior = NormalGammaPrior.from_std_parameters(
                np.random.multivariate_normal(mean, cov),
                np.ones(dim) * prior_count,
                ((1/variances) * prior_count),
                np.ones(dim) * prior_count
            )
        else:
            posterior = None

        return NormalDiagonalCovariance(prior, posterior)

    @staticmethod
    def sufficient_statistics(X):
        """Compute the sufficient statistics of the data.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
        return np.c_[X**2, X, np.ones_like(X), np.ones_like(X)]

    def __init__(self, prior, posterior=None):
        """Initialize the Bayesian normal distribution.

        Args:
            prior (``beer.priors.NormalGammaPrior``): Prior over the
                means and precisions.

        """
        self.prior = prior
        if posterior is not None:
            self.posterior = posterior
        else:
            self.posterior = copy.deepcopy(prior)

    @property
    def mean(self):
        np1, np2, _, _ = self.posterior.grad_lognorm().reshape(4, -1)
        return np2 / (-2 * np1)

    @property
    def cov(self):
        np1, _, _, _ = self.posterior.grad_lognorm().reshape(4, -1)
        return np.diag(1/(-2 * np1))

    def lognorm(self):
        """Expected value of the log-normalizer with respect to the
        posterior distribution of the parameters.

        Returns:
            float: expected value of the log-normalizer.

        """
        _, np2, np3, np4 = self.posterior.grad_lognorm().reshape(4, -1)
        return -np.sum(np3 + np4) - .5 * len(np2) * np.log(2 * np.pi)

    def accumulate_stats(self, X):
        """Accumulate the sufficient statistics to update the
        posterior distribution.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Accumulated sufficient statistics.

        """
        return self.sufficient_statistics(X).sum(axis=0)

    def expected_natural_params(self, T):
        '''Expected value of the natural parameters of the model given
        the sufficient statistics. 

        '''
        return self.posterior.grad_lognorm(), None

    def exp_llh(self, X, accumulate=False):
        """Expected value of the log-likelihood w.r.t to the posterior
        distribution over the parameters.

        Args:
            X (numpy.ndarray): Data as a matrix.
            accumulate (boolean): If True, returns the accumulated
                statistics.

        Returns:
            numpy.ndarray: Per-frame expected value of the
                log-likelihood.
            numpy.ndarray: Accumulated statistics (if ``accumulate=True``).

        """
        T = self.sufficient_statistics(X)
        exp_natural_params = self.posterior.grad_lognorm()

        # Note: the lognormalizer is already included in the expected
        # value of the natural parameters.
        exp_llh = T @ exp_natural_params

        if accumulate:
            acc_stats = T.sum(axis=0)
            return exp_llh, acc_stats

        return exp_llh

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


class NormalFullCovariance(ConjugateExponentialModel):
    """Bayesian Normal distribution with diagonal covariance matrix."""

    @staticmethod
    def create(dim, mean=None, cov=None, prior_count=1., random_init=False):
        if mean is None: mean = np.zeros(dim)
        if cov is None: cov = np.identity(dim)

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

    def __init__(self, prior, posterior=None):
        """Initialize the Bayesian normal distribution.

        Args:
            prior (``beer.priors.NormalGammaPrior``): Prior over the
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

    def lognorm(self):
        """Expected value of the log-normalizer with respect to the
        posterior distribution of the parameters.

        Returns:
            float: expected value of the log-normalizer.

        """
        grad = self.posterior.grad_lognorm()
        _, _, np3, np4, D = NormalWishartPrior.extract_natural_params(grad)
        return -np.sum(np3 + np4 - .5 * D * np.log(2 * np.pi))

    def accumulate_stats(self, X):
        """Accumulate the sufficient statistics to update the
        posterior distribution.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Accumulated sufficient statistics.

        """
        return self.sufficient_statistics(X).sum(axis=0)

    def exp_llh(self, X, accumulate=False):
        """Expected value of the log-likelihood w.r.t to the posterior
        distribution over the parameters.

        Args:
            X (numpy.ndarray): Data as a matrix.
            accumulate (boolean): If True, returns the accumulated
                statistics.

        Returns:
            numpy.ndarray: Per-frame expected value of the
                log-likelihood.
            numpy.ndarray: Accumulated statistics (if ``accumulate=True``).

        """
        T = self.sufficient_statistics(X)
        exp_natural_params = self.posterior.grad_lognorm()

        # Note: the lognormalizer is already included in the expected
        # value of the natural parameters.
        exp_llh = T @ exp_natural_params

        if accumulate:
            acc_stats = T.sum(axis=0)
            return exp_llh, acc_stats

        return exp_llh

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

