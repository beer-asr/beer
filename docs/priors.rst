Conjugate priors for the Bayesian models
========================================

The ``priors`` package contains all the conjugate priors used by the
Bayesian models.

    >>> from beer import priors
    >>> import numpy as np

Normal prior with isotropic covariance
--------------------------------------

Creation
^^^^^^^^

Creation from the natural parameters:

    >>> np1 = 6. * np.ones(2)
    >>> np2 = precision = np.ones(2) * 2.
    >>> prior = priors.NormalDiagonalCovariance(np1, np2)
    >>> prior.np1
    array([ 6.,  6.])
    >>> prior.np2
    array([ 2.,  2.])

Creation from the mean and variance parameters:

    >>> mean = np.ones(2) * 3
    >>> var = np.ones(2) * 0.5
    >>> prior = priors.NormalDiagonalCovariance.from_mean_variance(mean, var)
    >>> prior.np1
    array([ 6.,  6.])
    >>> prior.np2
    array([ 2.,  2.])

Creation from the mean and precision parameters:

    >>> mean = np.ones(2) * 3
    >>> precision = np.ones(2) * 2.
    >>> prior = priors.NormalDiagonalCovariance.from_mean_precision(mean,
    ...     precision)
    >>> prior.np1
    array([ 6.,  6.])
    >>> prior.np2
    array([ 2.,  2.])


Log-normalizer
^^^^^^^^^^^^^^

The log-normalization constant is computed according to the current
value of the parameters.

    >>> print(round(prior.lognorm(), 6))
    17.306853


Expectation of the sufficient statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> exp_t1, exp_t2 = prior.grad_lognorm()
    >>> exp_t1
    array([ 3.,  3.])
    >>> exp_t2
    array([-4.75, -4.75])


KL divergence
^^^^^^^^^^^^^

    >>> mean = np.ones(2) * 4
    >>> precision = np.ones(2)
    >>> prior2 = priors.NormalDiagonalCovariance.from_mean_precision(mean,
    ...     precision)
    >>> prior.kl_div(prior)
    0.0
    >>> prior2.kl_div(prior2)
    0.0
    >>> print(round(prior.kl_div(prior2), 6))
    1.193147
    >>> print(round(prior2.kl_div(prior), 6))
    2.306853

