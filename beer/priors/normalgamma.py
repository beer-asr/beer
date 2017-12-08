
"""NormalGamma conjugate prior of the Normal distribution."""

import numpy as np
from scipy.special import psi, gammaln


class NormalGammaPrior:
    """Multivariate distribution composed of K independent NormalGamma
    distribution.

    Standard Parameters:
        - mean: m0
        - precision: p0
        - shape: a0
        - rate: b0

    Sufficient statistics:
        - T_1(m, p) = -p/2
        - T_2(m, p) = p * m
        - T_3(m, p) = - (ln p) / 2
        - T_4(m, p) = (p * m^2) / 2

    Natural parameters:
        - np1 = p0 * m0^2 + 2 b0
        - np2 = p0 * m0
        - np3 = 2 * (a0 - 1/2)
        - np4 = p0

    """

    @classmethod
    def from_std_parameters(cls, means, precisions, shapes, rates):
        """Create a ``NormalGamma`` distribution from the standard
        (expectation) parameters.

        Args:
            means (numpy.ndarray): Mean for each dimension.
            precisions (numpy.ndarray): Precision for each dimension.
            shapes (numpy.ndarray): Shape for each dimension.
            rates (numpy.ndarray): Rate for each dimension.

        Returns
            ``NormalGamma``: An initialized ``NormalGamma``
                distribution.

        """
        return cls(
            precisions * (means**2) + 2 * rates,
            precisions * means,
            2 * (shapes - .5),
            precisions
        )


    def __init__(self, np1, np2, np3, np4):
        """Initialize the prior from its natural parameters.

        Args:
            np1 (numpy.ndarray): First natural parameter.
            np2 (numpy.ndarray): Second natural parameter.
            np3 (numpy.ndarray): Third natural parameter.
            np4 (numpy.ndarray): Fourth natural parameter.

        """
        self.np1 = np1
        self.np2 = np2
        self.np3 = np3
        self.np4 = np4

    def lognorm(self):
        """Log normalizer of the density.

        Returns:
            float: Log-normalization value for the current natural
                parameters.

        """
        lognorm = -.5 * np.log(self.np4) + gammaln(.5 * (self.np3 + 1))
        lognorm -= (.5 * (self.np3 + 1)) \
            * np.log(.5 * (self.np1 - (self.np2 ** 2) / self.np4))
        return lognorm

    def grad_lognorm(self):
        """Gradient of the log-normalizer. This correspond to the
        expected vlue of the sufficient statistics.

        Returns
            ``numpy.ndarray``: Expected value of the first sufficient
                statistics.
            ``numpy.ndarray``: Expected value of the second sufficient
                statistics.
            ``numpy.ndarray``: Expected value of the third sufficient
                statistics.
            ``numpy.ndarray``: Expected value of the fourth sufficient
                statistics.

        """
        exp_T1 = - (self.np3 + 1) / (2 * self.np1 - (self.np2 ** 2) / self.np4)
        exp_T2 =  ((self.np3 + 1) / 2) \
            * (1. / (2 * (self.np1 - (self.np2 ** 2) / self.np4))) \
            * (self.np2 / self.np4)
        exp_T3 = .5 * psi(.5 * (self.np3 + 1)) \
            - .5 * np.log(.5 * (self.np1 - (self.np2 ** 2) / self.np4))
        exp_T4 = -.5 * self.np4 - .5 * (self.np3 + 1) \
            * (1. / (2 * self.np1 - (self.np2 ** 2) / self.np4)) \
            * .5 * (self.np2 ** 2 ) / (self.np4 ** 2)

        return exp_T1, exp_T2, exp_T3, exp_T4

    def kl_div(self, other):
        """KL divergence between the two distribution of them form.

        Args:
            other (``JointGamma``): distribution with which to compute
                 the KL divergence.

        Returns:
            float: Value of the dirvergence.

        """
        exp_T1, exp_T2, exp_T3, exp_T4 = self.grad_lognorm()
        kl = (self.np1 - other.np1) @ exp_T1
        kl += (self.np2 - other.np2) @ exp_T2
        kl += (self.np3 - other.np3) @ exp_T3
        kl += (self.np4 - other.np4) @ exp_T4
        kl += other.lognorm() - self.lognorm()
        return kl

