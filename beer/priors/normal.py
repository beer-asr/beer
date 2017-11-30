
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

    def grad_lognorm(self):
        """Gradient of the log-normalizer. This correspond to the
        expected vlue of the sufficient statistics.

        Returns
            ``numpy.ndarray``: E [ T_1(x) ] = .
            ``numpy.ndarray``: Expected value of T_2(x) =

        """
        pass
