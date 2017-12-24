
"""Normal-Wishart conjugate prior of the Normal distribution."""

import numpy as np
from scipy.special import psi, gammaln


class NormalWishartPrior:
    """Normal-Wishart distribution conjugate prior of the multivariate
    Normal distribution.

    Standard Parameters:
        - mean (vector): m0
        - precision (scalar): p0
        - cov_inv_mean (matrix): V0
        - dof (scalar): v0

    Sufficient statistics:
        - T_1(m, p) = -1/2 * vec(P)
        - T_2(m, p) = P @ m
        - T_3(m, p) = -1/2 * trace(P @ m @ m^T)
        - T_4(m, p) = 1/2 * ln det(P)

    Natural parameters:
        - np1 = vec(p0 *  m0 @ m0^T + V0)
        - np2 = p0 * m0
        - np3 = p0
        - np4 = v0 - D

    D is the dimension of the Normal distribution.

    """

    @classmethod
    def from_std_parameters(cls, mean, precision, mean_precision, dof):
        """Create a ``NormalGamma`` distribution from the standard
        (expectation) parameters.

        Args:
            mean (numpy.ndarray): Mean for each dimension.
            precision (float): Precision of the Normal prior.
            mean_precision (numpy.ndarray): Mean of the precision
                 matrix.
            dof (float): Degree of freedom of the Wishart..

        Returns
            ``NormalWishart``: An initialized ``NormalWishart``
                distribution.

        """
        return cls(np.hstack([
            ((precision * np.outer(mean, mean)) + mean_precision).reshape(-1),
            precision *  mean,
            precision,
            dof - len(mean)
        ]))

    @staticmethod
    def extract_natural_params(natural_params):
        # We need to retrieve the 4 natural parameters organized as
        # follows:
        #   [ np1_1, ..., np1_D^2, np2_1, ..., np2_D, np3, np4]
        #
        # The dimension D is found by solving the polynomial:
        #   D^2 + D - len(self.natural_params[:-2]) = 0
        D = int(.5 * (-1 + np.sqrt(1 + 4 * len(natural_params[:-2]))))

        np1, np2 = natural_params[:int(D**2)].reshape(D, D), \
             natural_params[int(D**2):-2]
        np3, np4 = natural_params[-2:]
        return np1, np2, np3, np4, D

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
        np1, np2, np3, np4, D = NormalWishartPrior.extract_natural_params(
            self.natural_params)

        lognorm = .5 * ((np3 + D) * D * np.log(2) - D * np.log(np3))
        sign, logdet = np.linalg.slogdet(np1 - np.outer(np2, np2) / np3)
        lognorm += -.5 * (np4 + D) * sign * logdet
        lognorm += np.sum(gammaln(.5 * (np4 + D + 1 - np.arange(1, D + 1, 1))))
        return lognorm

    def grad_lognorm(self):
        """Gradient of the log-normalizer. This correspond to the
        expected value of the sufficient statistics.

        Returns
            ``numpy.ndarray``: Expected value of the sufficient
                statistics.

        """
        np1, np2, np3, np4, D = NormalWishartPrior.extract_natural_params(
            self.natural_params)

        outer = np.outer(np2, np2) / np3
        matrix = (np1 - outer)
        sign, logdet = np.linalg.slogdet(matrix)
        inv_matrix = np.linalg.inv(matrix)

        grad1 = -.5 * (np4 + D) * inv_matrix
        grad2 = (np4 + D) * inv_matrix @ (np2 / np3)
        grad3 = - D / (2 * np3) - .5 * (np4 + D) \
            * np.trace(inv_matrix @ (outer / np3))
        grad4 = .5 * np.sum(psi(.5 * (np4 + D + 1 - np.arange(1, D + 1, 1))))
        grad4 += -.5 * sign * logdet + .5 * D * np.log(2)
        return np.hstack([grad1.reshape(-1), grad2, grad3, grad4])

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

