
"""Normal distribution with prior over the mean and covariance matrix.

"""

from .. import priors
import copy
import numpy as np


class NormalDiagonalCovariance:
    """Bayesian Normal distribution with diagonal covariance matrix."""

    @staticmethod
    def create(dim):
        mean_prior = priors.NormalDiagonalCovariance.from_mean_precision(
            np.zeros(dim),
            np.ones(dim) * 1e-3
        )
        prec_prior = priors.JointGamma.from_shapes_rates(
            np.ones(dim) * 1e-3,
            np.ones(dim) * 1e-3
        )
        return NormalDiagonalCovariance(mean_prior, prec_prior)

    def __init__(self, mean_prior, precision_prior):
        """Initialize the Bayesian normal distribution.

        Args:
            mean_prior (beer.priors.NormalDiagonalCovariance): Prior
                over the mean parameters.
            precision_prior (beer.priors.JointGamma): Prior over the
                diagonal of the covariance matrix.

        """
        self.mean_prior = mean_prior
        self.precision_prior = precision_prior
        self.mean_posterior = copy.deepcopy(mean_prior)
        self.precision_posterior = copy.deepcopy(precision_prior)
        self._update()

    @property
    def np1(self):
        """First natural parameter vector."""
        return self._np1

    @property
    def np2(self):
        """Second natural parameter vector."""
        return self._np2

    def _update(self):
        self.exp_mean, self.exp_mean_quad = self.mean_posterior.grad_lognorm()
        self.exp_precision, self.exp_log_precision = \
            self.precision_posterior.grad_lognorm()
        self._np1 = self.exp_mean
        self._np2 = -.5 * self.exp_precision

    def lognorm(self):
        """Expected value of the log-normalizer with respect to the
        posterior distribution of the parameters.

        Returns:
            float: expected log-normalizer.

        """
        return .5 * (self.exp_mean_quad - self.exp_log_precision).sum()

    def accumulate_stats(self, X1, X2):
        """Accumulate the sufficient statistics (of the posterior) to
        update the posterior distribution.

        Args:
            X1 (numpy.ndarray): First sufficient statistics.
            X2 (numpy.ndarray): Second sufficient statistics.

        """
        nsamples = X1.shape[1]
        return {
            'mean_s1': self.exp_precision * X1.sum(axis=1),
            'mean_s2': self.exp_precision * nsamples,
            'prec_s1': self.exp_mean * X1.sum(axis=1) -.5 * X2.sum(axis=1) \
               + nsamples * self.exp_mean_quad,
            'prec_s2': .5 * nsamples
        }

    def exp_llh(self, X):
        """Expected value of the log-likelihood w.r.t to the posterior
        distribution over the parameters.

        Args:
            X (numpy.ndarray): Data as a matrix.

        Returns:
            numpy.ndarray: Per-frame expected value of the
                log-likelihood.
            dict: Accumulated statistics for the update.

        """
        X1, X2 = X, X**2
        exp_llh = self.np1 @ X1
        exp_llh += self.np2 @ X2
        exp_llh -= self.lognorm()
        exp_llh -= .5 * X.shape[0] * np.log(2 * np.pi)
        return exp_llh, self.accumulate_stats(X1, X2)

    def kl_div_posterior_prior(self):
        """KL divergence between the posterior and prior distribution.

        Returns:
            float: KL divergence.

        """
        kl_div = self.mean_posterior.kl_div(self.mean_prior)
        kl_div += self.precision_prior.kl_div(self.precision_prior)
        return kl_div

    def natural_grad_update(self, acc_stats, scale, lrate):
        """Perform a natural gradient update of the posteriors'
        parameters.

        Args:
            acc_stats (dict): Accumulated statistics.
            scale (float): Scale of the sufficient statistics.
            lrate (float): Learning rate.

        """
        # Compute the gradients.
        mean_grad_np1 = self.mean_prior.np1 + scale * acc_stats['mean_s1'] \
            - self.mean_posterior.np1
        mean_grad_np2 = self.mean_prior.np2 + scale * acc_stats['mean_s2'] \
            - self.mean_posterior.np2
        prec_grad_np1 = self.precision_prior.np1 + scale * acc_stats['prec_s1'] \
            - self.precision_posterior.np1
        prec_grad_np2 = self.precision_prior.np2 + scale * acc_stats['prec_s2'] \
            - self.precision_posterior.np2

        # Update the posterior distribution.
        self.mean_posterior.np1 += lrate * mean_grad_np1
        self.mean_posterior.np2 += lrate * mean_grad_np2
        self.precision_posterior.np1 += lrate * prec_grad_np1
        self.precision_posterior.np2 += lrate * prec_grad_np2

        self._update()

