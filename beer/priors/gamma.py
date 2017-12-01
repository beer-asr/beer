
"""Gamma conjugate prior."""

import numpy as np
from scipy.special import psi, gammaln


class JointGamma:
    """Multivariate distribution composed of K independent Gamma
    distribution.

    Standard Parameters:
        - shape: s
        - rate: r or scale: 1 / r

    Natural parameters:
        - np1 = -r
        - np2 = s - 1

    Sufficient statistics:
        - T_1(x) = x
        - T_2(x) = ln(x)

    """

    @classmethod
    def from_shapes_rates(cls, shapes, rates):
        """Create a ``JointGamma`` distribution from the ``shape`` and
        ``rate`` parameters.

        Args:
            shapes (numpy.ndarray): Shapes of the distribution for each
                dimension.
            rates (numpy.ndarray): Rates of the distribution for each
                dimension.

        Returns
            ``JointGamma``: An initialized ``JointGamma`` distribution.

        """
        return cls(-rates, shapes - 1)


    @classmethod
    def from_shapes_scales(cls, shapes, scales):
        """Create a ``JointGamma`` distribution from the ``shapes`` and
        ``scales`` parameters.

        Args:
            shapes (numpy.ndarray): Shapes of the distribution for each
                dimension.
            scales (numpy.ndarray): Scales of the distribution for each
                dimension.

        Returns
            ``JointGamma``: An initialized ``JointGamma`` distribution.

        """
        return cls(- 1. / scales, shapes - 1.)

    def __init__(self, np1, np2):
        """Initialize the prior from its natural parameters.

        Args:
            np1 (numpy.ndarray): First natural parameter.
            np2 (numpy.ndarray): Second natural parameter.

        """
        self.np1 = np1
        self.np2 = np2

    def lognorm(self):
        """Log normalizer of the density.

        Returns:
            float: Log-normalization value for the current natural
                parameters.

        """
        return np.sum(gammaln(self.np2 + 1) \
            - (self.np2 + 1) * np.log(-self.np1))

    def grad_lognorm(self):
        """Gradient of the log-normalizer. This correspond to the
        expected vlue of the sufficient statistics.

        Returns
            ``numpy.ndarray``: Expected value of the first sufficient
                statistics.
            ``numpy.ndarray``: Expected value of the second sufficient
                statistics.

        """
        return - (self.np2 + 1) / self.np1, \
            psi(self.np2 + 1) - np.log(-self.np1)

    def kl_div(self, other):
        """KL divergence between the two distribution of them form.

        Args:
            other (``JointGamma``): distribution with which to compute
                 the KL divergence.

        Returns:
            float: Value of the dirvergence.

        """
        exp_t1, exp_t2 = self.grad_lognorm()
        kl = (self.np1 - other.np1) @ exp_t1
        kl += (self. np2 - other.np2) @ exp_t2
        kl += other.lognorm() - self.lognorm()
        return kl

