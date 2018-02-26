
"""Dirichlet conjugate prior of the Categorical distribution."""

import numpy as np
from scipy.special import psi, gammaln, logsumexp


class DirichletPrior:
    """Dirichlet distribution conjugate prior of the categorical
    distribution.

    Standard Parameters:
        - beta (vector): b0

    Sufficient statistics:
        - T(pi) = log pi

    Natural parameters:
        - np = b0 - 1

    """

    @classmethod
    def from_std_parameters(cls, prior_counts):
        """Create a ``Dirichlet`` distribution from the standard
        (expectation) parameters.

        Args:
            prior_counts (numpy.ndarray): Prior count for each
                category.

        Returns
            ``Dirichlet``: An initialized ``Dirichlet`` distribution.

        """
        return cls(prior_counts - 1)

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
        return -gammaln(np.sum(self.natural_params + 1)) \
            + np.sum(gammaln(self.natural_params + 1))

    def grad_lognorm(self, normalize=False):
        """Gradient of the log-normalizer. This correspond to the
        expected value of the sufficient statistics.

        Args:
            normalize (boolean): Ensure the exponential of the results
                sum up to one.

        Returns
            ``numpy.ndarray``: Expected value of the sufficient
                statistics.

        """
        retval = -psi(np.sum(self.natural_params + 1)) \
            + psi(self.natural_params + 1)
        if normalize:
            retval -= logsumexp(retval)
        return retval

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
