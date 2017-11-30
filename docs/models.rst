Bayesian Models
===============

The ``models`` package contains all the Bayesian models.

    >>> from beer import models
    >>> import numpy as np

Bayesian Normal distribution  with diagonal covariance
------------------------------------------------------

Creation
^^^^^^^^

    >>> from beer import models
    >>> model = models.NormalDiagonalCovariance.create(dim=2)

Learning
^^^^^^^^

    >>> np.random.seed(1)
    >>> mean = np.ones(2) + 3.
    >>> cov = np.identity(2) * 1
    >>> data = np.random.multivariate_normal(mean, cov, size=1000).T
    >>> for i in range(10):
    ...     exp_llh, acc_stats = model.exp_llh(data)
    ...     model.natural_grad_update(acc_stats, scale=1.0, lrate=1.0)

    >>> model.exp_mean, model.exp_precision
    (array([ 4.03949772,  4.02663203]), array([ 0.98250549,  0.99250784]))

