
"""Normal density conjugate prior."""

import numpy as np


class NormalIsotropicCovariance:
    """Normal conjugate prior with isotropic covariance matrix.

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
        """Create a ``NormalPrior`` from mean and variance parameters.

        Args:
            mean (numpy.ndarray): Mean of the Normal density.
            variance (float): Variance of the Normal density.

        Returns
            ``NormalIsotropicCovariance``: An initialized Normal
                density.

        """
        return cls(mean / variance, 1. / variance)


    @classmethod
    def from_mean_precision(cls, mean, precision):
        """Create a ``NormalPrior`` from mean and precision parameters.

        Args:
            mean (numpy.ndarray): mean of the Normal density.
            precision (float): precision of the Normal density.

        Returns
            ``NormalIsotropicCovariance``: An initialized Normal
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

