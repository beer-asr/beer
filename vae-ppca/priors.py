import copy
import numpy as np
from scipy.special import gammaln, psi


class GammaPrior:
    """Joint Gamma distribution for the precision of the prior over the
    bases of the subspace H_W.

    Basic verifications:

    >>> prior = GammaPrior(shapes=np.ones(2), rates=np.ones(2))
    >>> prior = GammaPrior(shapes=-np.ones(2), rates=np.ones(2))
    Traceback (most recent call last):
        ...
    ValueError: "shapes" parameter should be an array of positive values
    >>> prior = GammaPrior(shapes=np.ones(2), rates=-np.ones(2))
    Traceback (most recent call last):
        ...
    ValueError: "rates" parameter should be an array of positive values

    """

    def __init__(self, shapes, rates):
        """Initialize the Gamma prior.

        >>> prior = GammaPrior(shapes=np.ones(2) + 2, rates=np.ones(2) + 1)
        >>> prior.np1
        array([-2., -2.])
        >>> prior.np2
        array([ 2.,  2.])

        """
        if shapes.min() <= 0:
            raise ValueError('"shapes" parameter should be an array '
                'of positive values')
        if rates.min() <= 0:
            raise ValueError('"rates" parameter should be an array '
                'of positive values')

        self.np1 = -rates
        self.np2 = shapes - 1

    def lognorm(self):
        """Log-normalizer function computed with the current parameters.

        >>> prior = GammaPrior(shapes=np.ones(2) + 2, rates=np.ones(2) + 1)
        >>> prior.lognorm()
        -2.7725887222397807

        """
        return -(self.np2 + 1) @ np.log(-self.np1) \
            + gammaln(self.np2 + 1).sum()

    def grad_lognorm(self):
        """Gradient of the log-normalizer w.r.t. the natural parameters.

        >>> prior = GammaPrior(shapes=np.ones(2) + 2, rates=np.ones(2) + 1)
        >>> prior.grad_lognorm()
        (array([ 1.5,  1.5]), array([ 0.22963715,  0.22963715]))

        """
        return -(self.np2 + 1) * (1/self.np1), \
            - np.log(-self.np1) + psi(self.np2 + 1)

    def kl_div(self, other):
        """ KL-Divergence between the prior and another Gamma
        distribution.

        >>> prior = GammaPrior(shapes=np.ones(2) + 2, rates=np.ones(2) + 1)
        >>> prior.kl_div(prior)
        0.0

        """
        grad1, grad2 = self.grad_lognorm()
        return (self.np1 - other.np1) @ grad1 \
            + (self.np2 - other.np2) @ grad2 \
            + other.lognorm() - self.lognorm()


class BasesPrior:
    """Prior over the bases of the subspace.

    """

    @staticmethod
    def create(dim, latent_dim, prec_shape=1., prec_rate=1.):
        """Create a prior over the bases of the subspace.

        >>> prior = BasesPrior.create(3, 2)
        >>> prior.np1
        array([[ 1.,  0.],
               [ 0.,  1.],
               [ 0.,  0.]])

        """
        prior_prec = GammaPrior(
            shapes=prec_shape * np.ones(latent_dim),
            rates=prec_rate * np.ones(latent_dim)
        )
        mean_bases = np.identity(dim)[:, :latent_dim]
        return BasesPrior(mean_bases, prior_prec)

    def __init__(self, mean_bases, prior_precision):
        """ Initialize the prior over the bases of the subpace. The
        prior is a compound between a Normal prior over the mean of
        the bases and a Gamma prior over the precision of the bases.

        >>> prior = BasesPrior.create(3, 2, prec_shape=2.)
        >>> prior.np1
        array([[ 2.,  0.],
               [ 0.,  2.],
               [ 0.,  0.]])

        """
        self.prior_precision = prior_precision
        self.posterior_precision = copy.deepcopy(prior_precision)
        self.exp_precision, _ = self.posterior_precision.grad_lognorm()
        self.np2 = self.exp_precision * np.identity(mean_bases.shape[1])
        self.np1 = mean_bases @ self.np2

    def update_hyperprior(self, exp_bases_quad):
        """Update the hyper-prior.

        >>> prior = BasesPrior.create(3, 2)
        >>> exp_bases_quad = np.array([[1., 0.], [0., 2.]])
        >>> prior.update_hyperprior(exp_bases_quad)
        >>> prior.posterior_precision.np1
        array([-1.5, -2. ])
        >>> prior.posterior_precision.np2
        array([ 1.5,  1.5])
        >>> prior.np2
        array([[ 1.66666667,  0.        ],
               [ 0.        ,  1.25      ]])

        """
        dim = self.np1.shape[0]
        latent_dim = self.np1.shape[1]
        self.posterior_precision.np1 = self.prior_precision.np1 \
            -.5 * np.diag(exp_bases_quad)
        self.posterior_precision.np2 = self.prior_precision.np2 + .5 * dim
        self.exp_precision, _ = self.posterior_precision.grad_lognorm()
        self.np2 = self.exp_precision * np.identity(latent_dim)


    def lognorm(self):
        """Log-normalizer function computed with the current parameters.

        Notes:
            The log-normalizer does not include the constant terms
            which can be either part of the log-normalizer or the
            log-base measure.

        >>> prior = BasesPrior.create(3, 2, prec_shape=2.)
        >>> prior.lognorm()
        0.61370563888010943

        """
        inv_np2 = np.linalg.inv(self.np2)
        sign, logdet = np.linalg.slogdet(self.np2)
        logdet *= sign
        return .5 * (np.trace(self.np1 @ inv_np2 @  self.np1.T) \
            - self.np1.shape[1] * logdet)

    def grad_lognorm(self):
        """Gradient of the log-normalizer w.r.t. the natural parameters.

        >>> prior = BasesPrior.create(3, 2, prec_shape=2.)
        >>> grad1, grad2 = prior.grad_lognorm()
        >>> grad1
        array([[ 1.,  0.],
               [ 0.,  1.],
               [ 0.,  0.]])
        >>> grad2
        array([[-1.25, -0.  ],
               [-0.  , -1.25]])

        """
        inv_np2 = np.linalg.inv(self.np2)
        grad1 = self.np1 @ inv_np2
        grad2 = (inv_np2 + grad1[:, :, None] * grad1[:, None, :]).sum(axis=0)

        return grad1, -.5 * grad2

    def kl_div(self, other):
        """ KL-Divergence between the prior and another BasesPrior.

        >>> prior_prec = GammaPrior(shapes=np.array([3., 4.]),
        ...                         rates=np.ones(2))
        >>> M = np.array([[1., 2.], [1., 2.]])
        >>> prior = BasesPrior(M, prior_prec)
        >>> prior.kl_div(prior)
        0.0

        """
        grad1, grad2 = self.grad_lognorm()
        return np.trace((self.np1 - other.np1).T @ grad1) \
            + np.trace((self.np2 - other.np2) @ grad2) \
            + other.lognorm() - self.lognorm()


class NormalPrior:
    """Normal prior.

    """

    def __init__(self, mean, precision):
        """Initialize the prior.

        >>> mean = np.zeros(2)
        >>> precision = np.identity(2)
        >>> prior = NormalPrior(mean, precision)
        >>> prior.np1
        array([ 0.,  0.])
        >>> prior.np2
        array([[ 1.,  0.],
               [ 0.,  1.]])

        """
        self.np1 = precision @ mean
        self.np2 = precision

    def lognorm(self):
        """Log-normalizer function computed with the current parameters.

        >>> mean = np.zeros(2)
        >>> precision = np.identity(2)
        >>> prior = NormalPrior(mean, precision)
        >>> prior.lognorm()
        0.0

        """
        dim = len(self.np1)
        mat_np2 = self.np2.reshape((dim, dim), order='F')
        inv_mat_np2 = np.linalg.inv(mat_np2)
        sign, logdet = np.linalg.slogdet(inv_mat_np2)
        logdet *= sign
        return .5 * (self.np1 @ inv_mat_np2 @ self.np1 - logdet)


    def grad_lognorm(self):
        """Gradient of the log-normalizer w.r.t. the natural parameters.

        >>> mean = np.zeros(2)
        >>> precision = np.identity(2)
        >>> prior = NormalPrior(mean, precision)
        >>> prior.grad_lognorm()
        (array([ 0.,  0.]), array([[-0.5, -0. ],
               [-0. , -0.5]]))

        """
        inv_np2 = np.linalg.inv(self.np2)
        grad_np1 = inv_np2 @ self.np1
        grad_np2 = inv_np2 @ self.np1[:, None] @ self.np1[None, :] \
            @ inv_np2 + inv_np2

        return grad_np1, -.5 * grad_np2

    def kl_div(self, other):
        """ KL-Divergence between the prior and another Normal
        distribution.

        >>> mean = np.zeros(2)
        >>> precision = np.identity(2)
        >>> prior = NormalPrior(mean, precision)
        >>> prior.kl_div(prior)
        0.0

        """
        grad1, grad2 = self.grad_lognorm()
        return (self.np1 - other.np1) @ grad1 \
            + np.trace((self.np2 - other.np2) @ grad2) \
            + other.lognorm() - self.lognorm()


class LatentPosterior:
    """ The set of posterior for each the latent variable.

    Note:
        This posterior distribution is specific as it is estimated on
        the fly during the training.

    """

    def __init__(self, np1, np2):
        """Initialize the posterior over the latent variables given its
        natural parameters.

        >>> np1 = np.array([[1., 2., 3.], [1., 2., 3.]])
        >>> np2 = 2 * np.identity(2)
        >>> posterior = LatentPosterior(np1, np2)
        >>> posterior.np1
        array([[ 1.,  2.,  3.],
               [ 1.,  2.,  3.]])
        >>> posterior.np2
        array([[ 2.,  0.],
               [ 0.,  2.]])

        """
        self.np1 = np1
        self.np2 = np2

    def lognorm(self):
        """Log-normalizer for each latent variable.

        >>> np1 = np.array([[1., 2., 3.], [1., 2., 3.]])
        >>> np2 = 3 * np.identity(2)
        >>> posterior = LatentPosterior(np1, np2)
        >>> posterior.lognorm()
        array([-0.76527896,  0.23472104,  1.90138771])

        """
        inv_np2 = np.linalg.inv(self.np2)
        sign, logdet = np.linalg.slogdet(self.np2)
        logdet *= sign
        return .5 * ((self.np1 * (inv_np2 @ self.np1)).sum(axis=0) \
            - logdet)

    def grad_lognorm(self):
        """Gradient of the log-normalizer for each of the latent
        variable.

        >>> np1 = np.array([[1., 2., 3.], [1., 2., 3.]])
        >>> np2 = 3 * np.identity(2)
        >>> posterior = LatentPosterior(np1, np2)
        >>> grad1, grad2 = posterior.grad_lognorm()
        >>> grad1, grad2
        (array([[ 0.33333333,  0.66666667,  1.        ],
               [ 0.33333333,  0.66666667,  1.        ]]), array([[[-0.22222222, -0.38888889, -0.66666667],
                [-0.05555556, -0.22222222, -0.5       ]],
        <BLANKLINE>
               [[-0.05555556, -0.22222222, -0.5       ],
                [-0.22222222, -0.38888889, -0.66666667]]]))

        """
        inv_np2 = np.linalg.inv(self.np2)
        grad1 = inv_np2 @ self.np1
        grad2 = grad1.T[:, :, None] * grad1.T[:, None, :] + inv_np2
        return grad1, -.5 * grad2.T

    def kl_div(self, prior):
        """Compute the KL divergence between each posterior
        distribution and the prior.

        >>> mean = np.zeros(2)
        >>> precision = np.identity(2)
        >>> prior = NormalPrior(mean, precision)
        >>> np1 = np.array([[0., 0., 0.], [0., 0., 0.]])
        >>> np2 = np.identity(2)
        >>> posterior = LatentPosterior(np1, np2)
        >>> posterior.kl_div(prior)
        array([ 0.,  0.,  0.])

        """
        dim = len(self.np2)
        grad1, grad2 = self.grad_lognorm()
        retval = ((self.np1 - prior.np1[:, None]) * grad1).sum(axis=0)

        # Compute the trace for each matrix (np2 - prior.np2) @ grad2
        matrices = (self.np2 - prior.np2) @ grad2
        retval += matrices[np.arange(dim), np.arange(dim), :].sum(axis=0)

        retval += prior.lognorm() - self.lognorm()

        return retval

if __name__ == '__main__':
    import doctest
    doctest.testmod()

