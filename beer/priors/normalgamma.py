
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
        - T_3(m, p) = -(p * m^2) / 2
        - T_4(m, p) = (ln p) / 2

    Natural parameters:
        - np1 = p0 * m0^2 + 2 * b0
        - np2 = p0 * m0
        - np3 = p0
        - np4 = 2 * a0 - 1

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
        return cls(np.hstack([
            precisions * (means**2) + 2 * rates,
            precisions * means,
            precisions,
            2 * shapes - 1
        ]))

    def __init__(self, natural_params):
        """Initialize the prior from its natural parameters.

        Args:
            natural_params (numpy.ndarray): Natural parameters.

        """
        self.natural_params = natural_params

    def lognorm(self):
        """Log normalizer of the density.

        Returns:
            float: Log-normalization value for the current natural
                parameters.

        """
        np1, np2, np3, np4 = self.natural_params.reshape(4, -1)
        lognorm = gammaln(.5 * (np4 + 1))
        lognorm += -.5 * np.log(np3)
        lognorm += -.5 * (np4 + 1) * np.log(.5 * (np1 - ((np2**2) / np3)))
        return lognorm.sum()

    def grad_lognorm(self):
        """Gradient of the log-normalizer. This correspond to the
        expected vlue of the sufficient statistics.

        Returns
            ``numpy.ndarray``: Expected value of the first sufficient

        """
        np1, np2, np3, np4 = self.natural_params.reshape(4, -1)
        grad1 = -(np4 + 1) / (2 * (np1 - ((np2 ** 2) / np3)))
        grad2 = (np2 * (np4 + 1)) / (np3 * np1 - (np2 ** 2))
        grad3 = - 1 / (2 * np3) - ((np2 ** 2) * (np4 + 1)) \
            / (2 * np3 * (np3 * np1 - (np2 ** 2)))
        grad4 = .5 * psi(.5 * (np4 + 1)) \
            - .5 *np.log(.5 * (np1 - ((np2 ** 2) / np3)))
        return np.hstack([grad1, grad2, grad3, grad4])

    def kl_div(self, other):
        """KL divergence between the two distribution of them form.

        Args:
            other (``JointGamma``): distribution with which to compute
                 the KL divergence.

        Returns:
            float: Value of the dirvergence.

        """
        return (self.natural_params - other.natural_params) @ self.grad_lognorm() \
            + other.lognorm() - self.lognorm()

