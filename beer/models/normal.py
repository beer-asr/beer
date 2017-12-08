
"""Normal distribution with prior over the mean and covariance matrix.

"""

from ..priors import NormalGammaPrior
import copy
import numpy as np
import torch
from torch.autograd import Variable


class NormalDiagonalCovariance:
    """Bayesian Normal distribution with diagonal covariance matrix."""

    @staticmethod
    def create(dim, means=None, precisions=None, shapes=None, rates=None):
        if means is None: means = np.zeros(dim)
        if precisions is None: precisions = np.ones(dim)
        if shapes is None: shapes = np.ones(dim)
        if rates is None: rates = np.ones(dim)

        prior = NormalGammaPrior.from_std_parameters(means, precisions,
                                                     shapes, rates)
        return NormalDiagonalCovariance(prior)

    def __init__(self, prior):
        """Initialize the Bayesian normal distribution.

        Args:
            prior (``beer.priors.NormalGammaPrior``): Prior over the
                means and precisions.

        """
        self.prior = prior
        self.posterior = copy.deepcopy(prior)

    def __call__(self, exp_x1, exp_x2):
        exp_T1, exp_T2, _, _ = self.posterior.grad_lognorm()
        return {
            'np_linear': Variable(torch.FloatTensor(exp_T2)),
            'np_quadr': Variable(torch.FloatTensor(exp_T1)),
            'exp_log_norm': Variable(torch.FloatTensor(np.array([self.lognorm()]))),
            'acc_stats': self.accumulate_stats(exp_x1.data.numpy().T,
                exp_x2.data.numpy().T)
        }


    def lognorm(self):
        """Expected value of the log-normalizer with respect to the
        posterior distribution of the parameters.

        Returns:
            float: expected value of the log-normalizer.

        """
        dim = len(self.posterior.np1)
        _, _, exp_T3, exp_T4 = self.posterior.grad_lognorm()
        return (exp_T3 @ np.ones(dim) + exp_T4 @ np.ones(dim))

    def accumulate_stats(self, X1, X2):
        """Accumulate the sufficient statistics (of the posterior) to
        update the posterior distribution.

        Args:
            X1 (numpy.ndarray): First sufficient statistics.
            X2 (numpy.ndarray): Second sufficient statistics.

        """
        dim = X1.shape[0]
        nsamples = X1.shape[1]
        return {
            'T1': X2.sum(axis=1),
            'T2': X1.sum(axis=1),
            'T3': -nsamples * np.ones(dim),
            'T4': -nsamples * np.ones(dim)
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
        np1, np2, _, _ = self.posterior.grad_lognorm()
        X1, X2 = X.T, X.T**2
        exp_llh = -np1 @ X2 - np2 @ X1 - self.lognorm()
        exp_llh -= .5 * X1.shape[0] * np.log(2 * np.pi)
        return exp_llh, self.accumulate_stats(X1, X2)

    def kl_div_posterior_prior(self):
        """KL divergence between the posterior and prior distribution.

        Returns:
            float: KL divergence.

        """
        return self.posterior.kl_div(self.prior)

    def natural_grad_update(self, state, scale, lrate):
        """Perform a natural gradient update of the posteriors'
        parameters.

        Args:
            acc_stats (dict): Accumulated statistics.
            scale (float): Scale of the sufficient statistics.
            lrate (float): Learning rate.

        """
        acc_stats = state['acc_stats']

        # Compute the gradients.
        grad_np1 = self.prior.np1 + scale * acc_stats['T1'] \
            - self.posterior.np1
        grad_np2 = self.prior.np2 + scale * acc_stats['T2'] \
            - self.posterior.np2
        grad_np3 = self.prior.np1 + scale * acc_stats['T3'] \
            - self.posterior.np3
        grad_np4 = self.prior.np2 + scale * acc_stats['T4'] \
            - self.posterior.np4

        # Update the posterior distribution.
        self.posterior.np1 += lrate * grad_np1
        self.posterior.np2 += lrate * grad_np2
        self.posterior.np3 += lrate * grad_np3
        self.posterior.np4 += lrate * grad_np4

