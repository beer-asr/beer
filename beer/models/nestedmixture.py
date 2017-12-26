
"""Bayesian Mixture model."""

from itertools import chain
from .model import ConjugateExponentialModel
from ..priors import DirichletPrior
from scipy.special import logsumexp
import copy
import numpy as np


class NestedMixture(ConjugateExponentialModel):
    """Bayesian NestedMixture Model."""

    @staticmethod
    def create(n_clusters, n_components_per_cluster, create_component_func,
               args, prior_count=1):
        # Create the prior over the weights of the global mixture.
        prior_weights = DirichletPrior.from_std_parameters(
            np.ones(n_clusters) * prior_count)

        # Created the weights of the nested mixtures.
        c_prior_weights = [DirichletPrior.from_std_parameters(
                           np.ones(n_components_per_cluster) * prior_count)
                           for i in range(n_clusters)]

        # Create the components of the nested mixture.
        c_components = [create_component_func(**args)
                        for i in range(n_clusters * n_components_per_cluster)]

        return NestedMixture(prior_weights, c_prior_weights, c_components)

    def __init__(self, prior_weights, c_prior_weights, c_components):
        """Initialize the Bayesian normal distribution.

        Args:
            prior_weights (``beer.priors.DirichletPrior``): Prior over
                the weights of the mixture.
            c_prior_weights (list of ``beer.priors.DirichletPrior``):
                Prior over weights of the nested mixture.
            c_components (list): List of components of the mixture.

        """
        # This will be initialize in the _prepare() call.
        self._np_params_matrix = None

        self.prior_weights = prior_weights
        self.posterior_weights = copy.deepcopy(prior_weights)
        self.c_prior_weights = c_prior_weights
        self.c_posterior_weights = copy.deepcopy(c_prior_weights)
        self.c_components = c_components

        self._prepare()

    @property
    def weights(self):
        """Expected value of the weights."""
        return np.exp(self.posterior_weights.grad_lognorm(normalize=True))

    def sufficient_statistics(self, X):
        """Compute the sufficient statistics of the data.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
        return np.c_[self.c_components[0].sufficient_statistics(X),
                     np.ones(X.shape[0])]

    def _prepare(self):
        matrix = np.vstack([component.posterior.grad_lognorm()
            for component in self.c_components])
        #log_weights = self.posterior_weights.grad_lognorm()
        log_weights = np.zeros(len(self.c_posterior_weights))
        c_log_weights = np.hstack([log_weights[i] + post_weights.grad_lognorm()
            for i, post_weights in enumerate(self.c_posterior_weights)])
        self._np_params_matrix = np.c_[matrix, c_log_weights]

    def expected_natural_params(self, mean, var):
        '''Expected value of the natural parameters of the model given
        the sufficient statistics.

        '''
        T = self.components[0].sufficient_statistics_from_mean_var(mean, var)
        T2 = np.c_[T, np.ones(T.shape[0])]

        # Inference.
        per_component_exp_llh = T2 @ self._np_params_matrix.T
        exp_llh = logsumexp(per_component_exp_llh, axis=1)
        resps = np.exp(per_component_exp_llh - exp_llh[:, None])

        # Build the matrix of expected natural parameters.
        matrix = np.c_[[component.expected_natural_params(mean, var)[0][0]
                        for component in self.components]]

        # Accumulate the sufficient statistics.
        acc_stats = resps.T @ T2[:, :-1], resps.sum(axis=0)

        return (resps @ matrix), acc_stats

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
        # Get the sufficient statistics.
        T = self.sufficient_statistics(X)

        # Per cluster and per component expected log-likelihood.
        per_cluster_component_exp_llh = T @ self._np_params_matrix.T

        # Re-organize the exp. log-likelihood into a N x C x K matrix
        # where N is the number of frames, C is the number of
        # components per cluster and K is the number of clusters.
        M = per_cluster_component_exp_llh.reshape(T.shape[0], -1,
            len(self.c_posterior_weights)) \
            + self.posterior_weights.grad_lognorm()

        # Conditional responsibilities of the components.
        M_lognorm = logsumexp(M, axis=1)
        w_M_lognorm = M_lognorm
#+ self.posterior_weights.grad_lognorm()

        # Total log-likelihood and responsibilities of the clusters.
        exp_llh = logsumexp(w_M_lognorm, axis=1)
        resps = np.exp(w_M_lognorm - exp_llh[:, None])

        if accumulate:
            c_resps = np.exp(M - M_lognorm[:, None, :])
            acc_stats1 = resps.sum(axis=0)
            matrix = c_resps * resps[:, None, :]
            acc_stats2 = matrix.reshape(matrix.shape[0], matrix.shape[2],
                -1).sum(axis=0)
            acc_stats3 = matrix.reshape(X.shape[0], -1).T @ T[:, :-1]
            return exp_llh, (acc_stats1, acc_stats2, acc_stats3)

        return exp_llh

    def kl_div_posterior_prior(self):
        """KL divergence between the posterior and prior distribution.

        Returns:
            float: KL divergence.

        """
        kl = self.posterior_weights.kl_div(self.prior_weights)
        for prior, posterior in zip(self.c_prior_weights,
                                    self.c_posterior_weights):
            kl += posterior.kl_div(prior)
        for component in self.c_components:
            kl += component.posterior.kl_div(component.prior)
        return kl

    def natural_grad_update(self, acc_stats, scale, lrate):
        """Perform a natural gradient update of the posteriors'
        parameters.

        Args:
            acc_stats (tuple): Accumulated statistics.
            scale (float): Scale of the sufficient statistics.
            lrate (float): Learning rate.

        """
        acc_stats1, acc_stats2, acc_stats3 = acc_stats

        # Update the components.
        for i, component in enumerate(self.c_components):
            # Compute the natural gradient.
            natural_grad = component.prior.natural_params \
                + scale * acc_stats3[i] \
                - component.posterior.natural_params

            # Update the posterior distribution.
            component.posterior.natural_params += lrate * natural_grad

        # Update the cluster weights.
        for i, data in enumerate(zip(self.c_prior_weights,
                                     self.c_posterior_weights)):
            prior, post = data
            # Compute the natural gradient.
            natural_grad = prior.natural_params \
                + scale * acc_stats2[i] - post.natural_params

            # Update the posterior distribution.
            post.natural_params += lrate * natural_grad

        # Update the weights.
        natural_grad = self.prior_weights.natural_params \
            + scale * acc_stats1 - self.posterior_weights.natural_params
        self.posterior_weights.natural_params += lrate * natural_grad

        self._prepare()

