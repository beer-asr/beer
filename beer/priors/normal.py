
"""Normal density conjugate prior."""

import numpy as np


class NormalDiagonalCovariance:
    """Normal conjugate prior with a diagonal covariance matrix.

    Standard Parameters:
        - mean: m
        - precision: p

    Natural parameters:
        - np1 = m * p
        - np2 = p

    Sufficient statistics:
        - T_1(x) = x
        - T_2(x) = -0.5 * x^2

    """

    @classmethod
    def from_mean_variance(cls, mean, variance):
        """Create a Normal density from mean and variance parameters.

        Args:
            mean (numpy.ndarray): Mean of the Normal density.
            variance (float): Variance of the Normal density.

        Returns
            ``NormalDiagonalCovariance``: An initialized Normal
                density.

        """
        return cls(mean / variance, 1. / variance)


    @classmethod
    def from_mean_precision(cls, mean, precision):
        """Create a Normal density from mean and precision parameters.

        Args:
            mean (numpy.ndarray): mean of the Normal density.
            nit__(self, np1, np2):precision (flo): precision of the Normal density.

        Returns
            ``NormalDiagonalCovariance``: An initialized Normal
                density.

        """
        return cls(precision * mean, precision)

    def __init__(self, np1, np2):
        """Initialize the prior from its natural parameters.

        Args:
            np1 (numpy.ndarray): First natural parameter (p * m).
            np2 (numpy.ndarray): Second natural parameter (p).

        """
        self.np1 = np1
        self.np2 = np2

    def lognorm(self):
        """Log normalizer of the density.

        Returns:
            float: Log-normalization value for the current natural
                parameters.

        """
        return .5 * ((self.np1**2 / self.np2) - np.log(self.np2)).sum()

    def grad_lognorm(self):
        """Gradient of the log-normalizer. This correspond to the
        expected vlue of the sufficient statistics.

        Returns
            ``numpy.ndarray``: Expected value of the first sufficient
                statistics.
            ``numpy.ndarray``: E [ -0.5 * x^2 ] = Expected value of the second
                sufficient statistics.

        """
        return self.np1 / self.np2, -.5 * ((self.np1 / self.np2)**2 \
            + 1. / self.np2)

    def kl_div(self, other):
        """KL divergence between the to Normal density with diagonal
        prior.

        Args:
            other (``NormalDiagonalCovaraince): density with which to
                compute the KL divergence.

        Returns:
            float: Value of the dirvergence.

        """
        exp_t1, exp_t2 = self.grad_lognorm()
        kl = (self.np1 - other.np1) @ exp_t1
        kl += (self. np2 - other.np2) @ exp_t2
        kl += other.lognorm() - self.lognorm()
        return kl

