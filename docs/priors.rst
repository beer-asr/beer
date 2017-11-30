Conjugate priors for the Bayesian models
========================================

The ``priors`` package contains all the conjugate priors used by the
Bayesian models.

    >>> from beer import priors
    >>> import numpy as np

Normal prior with isotropic covariance
--------------------------------------

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

