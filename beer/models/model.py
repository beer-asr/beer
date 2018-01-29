
'''Abstract Base Class for a model.'''

import abc
import numpy as np


class Model(metaclass=abc.ABCMeta):
    """Abstract Base Class for all beer's model."""

    @abc.abstractmethod
    def _fit_step(self, mini_batch):
        NotImplemented


class ConjugateExponentialModel(metaclass=abc.ABCMeta):
    '''Abstract base class for Conjugate Exponential models.'''


    @abc.abstractmethod
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
        NotImplemented


    @abc.abstractmethod
    def kl_div_posterior_prior(self):
        """KL divergence between the posterior and prior distribution.

        Returns:
            float: KL divergence.

        """
        NotImplemented

    @abc.abstractmethod
    def natural_grad_update(self, acc_stats, scale, lrate):
        """Perform a natural gradient update of the posteriors'
        parameters.

        Args:
            acc_stats (dict): Accumulated statistics.
            scale (float): Scale of the sufficient statistics.
            lrate (float): Learning rate.

        """
        NotImplemented
