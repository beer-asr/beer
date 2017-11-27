
"""Probabilistic Principal Components Analysis (PPCA) model.

"""

import copy
import numpy as np
import priors


class PPCA:
    """PPCA model."""

    @staticmethod
    def create(dim, latent_dim=None, mean=None, mean_precision=1.,
               precision_shape=1., precision_rate=1.,
               precision_bases_shape=1., precision_bases_rate=1.):
        """Create a PPCA model.

        >>> model = PPCA.create(2)

        """

        if not latent_dim:
            latent_dim = dim
        if not mean:
            mean = np.zeros(dim)

        # Prior over the latent variables.
        prior_latent = priors.NormalPrior(
            mean=np.zeros(latent_dim),
            precision=np.identity(latent_dim)
        )

        # Prior over the global mean of the model.
        prior_mean = priors.NormalPrior(
            mean=np.zeros(dim),
            precision=mean_precision * np.identity(dim)
        )

        # Prior over the global precision of the model.
        prior_prec = priors.GammaPrior(
            shapes=precision_shape * np.ones(1),
            rates=precision_rate * np.ones(1)
        )

        # Hierarchical prior over the bases of the subspace.
        prior_prec_bases = priors.GammaPrior(
            shapes=precision_bases_shape * np.ones(latent_dim),
            rates=precision_bases_rate * np.ones(latent_dim)
        )
        prior_bases = priors.BasesPrior(
            mean_bases=np.zeros((dim, latent_dim)),
            prior_precision=prior_prec_bases
        )

        model = PPCA(prior_latent, prior_mean, prior_prec, prior_bases)

        return model


    def __init__(self, prior_latent, prior_mean, prior_precision, prior_bases):
        """Initialize the PPCA model.

        >>> mean = np.zeros(2)
        >>> precision = np.identity(2)
        >>> prior_latent = priors.NormalPrior(mean, precision)
        >>> prior_mean = priors.NormalPrior(mean, precision)
        >>> prior_prec = priors.GammaPrior(shapes=np.ones(1),
        ...                                rates=np.ones(1))
        >>> prior_prec_bases = priors.GammaPrior(shapes=np.ones(2),
        ...                                      rates=np.ones(2))
        >>> mean_bases = np.zeros((2, 2))
        >>> prior_bases = priors.BasesPrior(mean_bases, prior_prec_bases)
        >>> model = PPCA(prior_latent, prior_mean, prior_prec, prior_bases)

        """
        self.prior_latent = prior_latent
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.prior_bases = prior_bases

        # Initialize the variational posterior distributions.
        self.posterior_mean = copy.deepcopy(prior_mean)
        self.posterior_precision = copy.deepcopy(prior_precision)

        # Set the initial bases of the subspace.
        dim = len(self.prior_mean.np1)
        latent_dim = self.prior_bases.np1.shape[1]
        initial_bases = np.identity(dim)[:, :latent_dim]
        self.posterior_bases = priors.BasesPrior(initial_bases,
            self.prior_bases.prior_precision)

        self.update()

    def update(self):
        """Update the expected value of the natural parameters.

        >>> model = PPCA.create(3, 2)
        >>> model.exp_precision
        array([ 1.])
        >>> model.exp_log_precision
        array([-0.57721566])
        >>> model.exp_mean
        array([ 0.,  0.,  0.])
        >>> model.exp_mean_quad
        array([ 1.,  1.,  1.])
        >>> model.exp_bases
        array([[ 1.,  0.],
               [ 0.,  1.],
               [ 0.,  0.]])
        >>> model.exp_bases_quad
        array([[ 4.,  0.],
               [ 0.,  4.]])

        """
        self.exp_precision, self.exp_log_precision = \
            self.posterior_precision.grad_lognorm()

        self.exp_mean, tmp_matrix = \
            self.posterior_mean.grad_lognorm()
        self.exp_mean_quad = -2 * np.diag(tmp_matrix)

        self.exp_bases, tmp_matrix = \
            self.posterior_bases.grad_lognorm()
        self.exp_bases_quad = -2 * tmp_matrix

    def latent_posteriors(self, X):
        """Estimate the posterior distribution of the latent variables
        given the data.

        >>> model = PPCA.create(3, 2)
        >>> model.exp_mean = np.array([1. , 1., 1.])
        >>> X = np.array([[1., 2., 3., 4.],
        ...     [1., 2., 3., 4.],
        ...     [1., 2., 3., 4.]])
        >>> posteriors = model.latent_posteriors(X)
        >>> posteriors.np1
        array([[ 0.,  1.,  2.,  3.],
               [ 0.,  1.,  2.,  3.]])
        >>> posteriors.np2
        array([[ 5.,  0.],
               [ 0.,  5.]])
        >>> grad1, grad2 = posteriors.grad_lognorm()
        >>> grad1
        array([[ 0. ,  0.2,  0.4,  0.6],
               [ 0. ,  0.2,  0.4,  0.6]])
        >>> -2 * grad2.T
        array([[[ 0.2 ,  0.  ],
                [ 0.  ,  0.2 ]],
        <BLANKLINE>
               [[ 0.24,  0.04],
                [ 0.04,  0.24]],
        <BLANKLINE>
               [[ 0.36,  0.16],
                [ 0.16,  0.36]],
        <BLANKLINE>
               [[ 0.56,  0.36],
                [ 0.36,  0.56]]])

        """
        np1 = self.prior_latent.np1[:, None] + self.exp_precision * (
            self.exp_bases.T @ (X - self.exp_mean[:, None]))
        np2 = self.prior_latent.np2 + self.exp_precision *  \
            self.exp_bases_quad
        return priors.LatentPosterior(np1,  np2)

    def log_base_measure(self, X):
        """Log base measure of the model given the data.

        >>> model = PPCA.create(3, 2)
        >>> X = np.ones((3, 4))
        >>> model.log_base_measure(X)
        -11.027262398456072

        """
        return -.5 * (X.shape[0] * X.shape[1]) * np.log(2 * np.pi)

    def kernel(self, X, latent_posteriors, return_exp_latent=False):
        """The "kernel" of the Gaussian. By "kernel" we mean the
        expected value of the quadratic function of the Gaussian
        density (not including the precision) i.e.:

            -.5 * \sum (x_n - (W * z_n - m) )^T * (x_n - (W * z_n - m))

        >>> model = PPCA.create(3, 2)
        >>> X = np.array([[1., 2., 3., 4.],
        ...     [1., 2., 3., 4.],
        ...     [1., 2., 3., 4.]])
        >>> posteriors = model.latent_posteriors(X)
        >>> model.kernel(X, posteriors)
        array([ -3.56,  -7.34, -13.64, -22.46])

        """
        exp_latent, exp_latent_quad = latent_posteriors.grad_lognorm()
        exp_latent_quad *= -2

        N = X.shape[1]
        dim = X.shape[0]
        latent_dim = exp_latent.shape[0]

        # -.5 * E[ Trace( m * m^T ) ]
        kernel = -.5 * self.exp_mean_quad.sum()

        # -.5 * x_n^T x_n
        kernel += -.5 * (X**2).sum(axis=0)

        #  E[ ( W * z_n + m )^T x_n ]
        kernel += (((self.exp_bases @ exp_latent) + self.exp_mean[:, None]) \
            * X).sum(axis=0)

        #  E[ m^T W z_n  ]
        kernel += - ((self.exp_bases @ exp_latent) \
            * self.exp_mean[:, None]).sum(axis=0)

        # E[ Trace(W^T * W * z_n * z_n^T) ]
        matrices = (self.exp_bases_quad @ exp_latent_quad)
        kernel += -.5 * matrices[np.arange(latent_dim),
            np.arange(latent_dim), :].sum(axis=0)

        if return_exp_latent:
            return kernel, exp_latent, exp_latent_quad
        return kernel

    def exp_llh(self, X, accumulate_stats=False):
        """Expected value of the log-likelihood.

        >>> model = PPCA.create(3, 2)
        >>> X = np.array([[1., 2., 3., 4.],
        ...     [1., 2., 3., 4.],
        ...     [1., 2., 3., 4.]])
        >>> llh, acc_stats = model.exp_llh(X, accumulate_stats=True)
        >>> print(round(llh, 6))
        -61.490556
        >>> acc_stats['mean_s1']
        array([  8.,   8.,  10.])
        >>> acc_stats['mean_s2']
        array([[ 4.,  0.,  0.],
               [ 0.,  4.,  0.],
               [ 0.,  0.,  4.]])
        >>> acc_stats['precision_s1']
        -47.0
        >>> acc_stats['precision_s2']
        6.0
        >>> acc_stats['bases_s1']
        array([[ 6.,  6.],
               [ 6.,  6.],
               [ 6.,  6.]])
        >>> acc_stats['bases_s2']
        array([[ 2. ,  1.2],
               [ 1.2,  2. ]])

        """
        N = X.shape[1]
        dim = X.shape[0]

        # Estimate the posterior distribution for each latent variable.
        posteriors = self.latent_posteriors(X)

        # Estimate the kernel for each data point.
        kernels, exp_latent, exp_latent_quad = self.kernel(X, posteriors,
            return_exp_latent=True)

        # Compute the total lower bound.
        exp_llh = self.log_base_measure(X) \
            + .5 * N * dim * self.exp_log_precision \
            + self.exp_precision * kernels.sum()

        # Accumulate the sufficient statistics if requested.
        if accumulate_stats:
            acc_mean = X.sum(axis=-1)
            acc_mean -= self.exp_bases @ exp_latent.sum(axis=-1)
            acc_mean *= self.exp_precision

            acc_precision = kernels.sum()
            acc_precision = min(acc_precision, -.5 * N * dim * 1e-2)

            X_mean = X - self.exp_mean[:, None]
            acc_bases = (X_mean[:, :] @ exp_latent.T[:, :])
            acc_bases *= self.exp_precision
            stats = {
                'mean_s1': acc_mean,
                'mean_s2': self.exp_precision * N * np.identity(dim),
                'precision_s1': acc_precision,
                'precision_s2': .5 * N * dim,
                'bases_s1': acc_bases,
                'bases_s2': self.exp_precision * exp_latent_quad.sum(axis=-1)
            }
            return exp_llh[0], stats

        return exp_llh[0]

    def kl_div(self, X):
        posteriors = self.latent_posteriors(X)
        KL_latent = posteriors.kl_div(self.prior_latent).sum()
        KL = 0
        KL += self.posterior_precision.kl_div(self.prior_precision)
        KL += self.posterior_mean.kl_div(self.prior_mean)
        KL += self.posterior_bases.kl_div(self.prior_bases)
        return KL_latent, KL


    def natural_gradient_update(self, acc_stats, scale, lrate=1.):
        """Update the parameters of the posteriors following a natural
        gradient ascent scheme.

        >>> model = PPCA.create(3, 2)
        >>> X = np.array([[1., 2., 3., 4.],
        ...     [1., 2., 3., 4.],
        ...     [1., 2., 3., 4.]])
        >>> X = np.ones((3, 1000))
        >>> llh, acc_stats = model.exp_llh(X, accumulate_stats=True)
        >>> model.natural_gradient_update(acc_stats, 1.)
        >>> model.posterior_precision.np1
        array([-3561.])
        >>> model.posterior_precision.np2
        array([ 1500.])
        >>> model.exp_precision
        array([ 0.42151081])

        """
        ngrad_mean_np1 = self.prior_mean.np1 + scale * acc_stats['mean_s1'] \
            - self.posterior_mean.np1
        ngrad_mean_np2 = self.prior_mean.np2 + scale * acc_stats['mean_s2'] \
            - self.posterior_mean.np2
        ngrad_bases_np1 = self.prior_bases.np1 + scale * acc_stats['bases_s1'] \
            - self.posterior_bases.np1
        ngrad_bases_np2 = self.prior_bases.np2 + scale * acc_stats['bases_s2'] \
            - self.posterior_bases.np2
        ngrad_precision_np1 = self.prior_precision.np1 \
            + scale * acc_stats['precision_s1'] - self.posterior_precision.np1
        ngrad_precision_np2 = self.prior_precision.np2 \
            + scale * acc_stats['precision_s2'] - self.posterior_precision.np2

        self.posterior_mean.np1 += lrate * ngrad_mean_np1
        self.posterior_mean.np2 += lrate * ngrad_mean_np2
        self.posterior_bases.np1 += lrate * ngrad_bases_np1
        self.posterior_bases.np2 += lrate * ngrad_bases_np2
        self.posterior_precision.np1 += lrate * ngrad_precision_np1
        self.posterior_precision.np2 += lrate * ngrad_precision_np2

        # This will compute the expected value of the parameters with
        # respect to the posterior distributions.
        self.update()

        # Update the hyper-prior of the prior over the bases. It has
        # to be called after self.update() as it requires the expected
        # value of the quadratic product of the bases.
        self.prior_bases.update_hyperprior(self.exp_bases_quad)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

