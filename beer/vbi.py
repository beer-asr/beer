
'Variational Bayes Inference.'


import torch


def add_acc_stats(acc_stats1, acc_stats2):
    '''Add two ditionary of accumulated statistics. Both dictionaries
    may have different set of keys. The elements in the dictionary
    should implement the sum operation.

    Args:
        acc_stats1 (dict): First set of accumulated statistics.
        acc_stats2 (dict): Second set of accumulated statistics.

    Returns:
        dict: `acc_stats1` + `acc_stats2`

    '''
    keys1, keys2 = set(acc_stats1.keys()), set(acc_stats2.keys())
    new_stats = {}
    for key in keys1.intersection(keys2):
        new_stats[key] = acc_stats1[key] + acc_stats2[key]

    for key in keys1.difference(keys2):
        new_stats[key] = acc_stats1[key]

    for key in keys2.difference(keys1):
        new_stats[key] = acc_stats2[key]

    return new_stats


def scale_acc_stats(acc_stats, scale):
    '''Scale a set of sufficient statistics.

    Args:
        acc_stats (dict): Accumulated sufficient statistics.
        scale (float): Scaling factor.

    Returns:
        dict: Scaled accumulated sufficient statistics.

    '''
    new_stats = {}
    for key, val in acc_stats.items():
        new_stats[key] = scale * val
    return new_stats


class EvidenceLowerBoundInstance:
    '''Evidence Lower Bound of a data set given a model.

    Note:
        This object should not be created directly.

    '''

    def __init__(self, elbo_value, acc_stats, model_parameters, minibatchsize,
                 datasize):
        self._elbo_value = elbo_value
        self._acc_stats = acc_stats
        self._model_parameters = set(model_parameters)
        self._minibatchsize = minibatchsize
        self._datasize = datasize

    def __str__(self):
        return str(self._elbo_value)

    def __float__(self):
        return float(self._elbo_value)

    def __add__(self, other):
        if not isinstance(other, EvidenceLowerBoundInstance):
            raise ValueError('EvidenceLowerBoundInstance')
        if self._datasize != other._datasize:
            raise ValueError('Cannot add ELBOs evaluated on different data set')

        return EvidenceLowerBoundInstance(
            self._elbo_value + other._elbo_value,
            add_acc_stats(self._acc_stats, other._acc_stats),
            self._model_parameters.union(other._model_parameters),
            self._minibatchsize + other._minibatchsize,
            self._datasize
        )

    def backward(self):
        '''Compute the gradient of the loss w.r.t. to standard
        ``pytorch`` parameters.
        '''
        # Pytorch minimizes the loss ! We change the sign of the ELBO
        # just before to compute the gradient.
        (-self._elbo_value).backward()

    def natural_backward(self):
        '''Compute the natural gradient of the loss w.r.t. to all the
        :any:`BayesianParameter`.
        '''
        scale = self._datasize / self._minibatchsize
        for parameter in self._model_parameters:
            acc_stats = self._acc_stats[parameter]
            parameter.accumulate_natural_grad(scale * acc_stats)


def evidence_lower_bound(model=None, minibatch_data=None, datasize=-1,
                         fast_eval=False, **kwargs):
    '''Evidence Lower Bound objective function of Variational Bayes
    Inference.

    If no `model` and no `data` argument are provided but the
    `datasize` argument is given, the function return an "empty"
    ``EvidenceLowerBoundInstance`` object which can be used to
    initialize the accumulatation of several
    ``EvidenceLowerBoundInstance`.

    Args:
        model (:any:`BayesianModel`): The Bayesian model with which to
            compute the ELBO.
        minibatch_data (``torch.Tensor``): Data of the minibatch on
            which to evaluate the ELBO.
        datasize (int): Number of data points of the total training
            data. If set to 0 or negative values, the size of the
            provided `minibatch_data` will be used instead.
        fast_eval (boolean): If true, skip computing KL-divergence for the
            global parameters.
        kwargs (object): Model specific extra parameters to evalute the
            ELBO.

    Returns:
        ``EvidenceLowerBoundInstance``

    Example:
        >>> # Assume X is our data set and "model" is the model to be
        >>> # trained.
        >>> elbo_fn = beer.EvidenceLowerBound(len(X))
        >>> elbo = beer.evidence_lower_bound(model, X)
        ...
        >>> # Compute gradient of the Baysian parameters.
        >>> elbo.natural_backward()
        ...
        >>> # Compute gradient of standard pytorch parameters.
        >>> elbo.backward()
        ...
        >>> round(float(elbo), 3)
        >>> -10.983


    Note:
        Practically speaking, ``beer`` implements a stochastic version
        of the traditional ELBO. This allows to do stochastic training
        of the models with small batches. It is therefore necessary
        to provide the total length (in frames) of the data when
        creating the loss function as it will scale the natural
        gradient accordingly.

    '''
    if model is None and  minibatch_data is None and datasize > 0:
        return EvidenceLowerBoundInstance(0., {}, [], 0, datasize)
    elif model is None or minibatch_data is None:
        raise ValueError('if datasize is not provided, need at least "model" '
                         'and "minibatch_data"')

    mb_datasize = len(minibatch_data)
    if datasize <= 0:
        datasize = mb_datasize

    # Estimate the scaling constant of the stochastic ELBO.
    scale = datasize / float(mb_datasize)

    # Compute the ELBO.
    stats = model.sufficient_statistics(minibatch_data)
    exp_llh = model(stats, **kwargs)
    if not fast_eval:
        kl_div = model.kl_div_posterior_prior().sum()
    else:
        kl_div = 0.
    elbo_value = scale * exp_llh.sum() - kl_div

    # Accumulate the statistics and scale them accordingly.
    acc_stats = model.accumulate(stats)

    # Clean up intermediary results.
    model.clear_cache()

    return EvidenceLowerBoundInstance(elbo_value, acc_stats,
                                      model.bayesian_parameters(),
                                      mb_datasize, datasize)


class BayesianModelOptimizer:
    '''Generic optimizer for :any:`BayesianModel` subclasses.

    Args:
        parameters (list): List of :any:`BayesianParameter`.
        lrate (float): Learning rate for the :any:`BayesianParameter`.
        std_optim (``torch.Optimizer``): pytorch optimizer.

    Note:
        For models that require some standard gradient descent (for
        instance Variational AutoEncoder), it is possible to combined
        natural and standard gradient descent by providing a pytorch
        optimizer through the keyword argument ``std_optim``.

    Example:
        >>> # Assume "model" is a BayesianModel to be trained and X is
        >>> # the dataset.
        >>> elbo_fn = beer.EvidenceLowerBound(len(X))
        >>> optim = beer.BayesianModelOptimizer(model.parameters)
        >>> for epoch in range(10):
        >>>     optim.zero_grad()
        >>>     elbo = elbo_fn(model, X)
        >>>     elbo.natural_backward()
        >>>     optim.step()

    '''

    def __init__(self, parameters, lrate=1., std_optim=None):
        '''
        Args:
            parameters (list): List of ``BayesianParameters``.
            lrate (float): learning rate.
            std_optim (``torch.optim.Optimizer``): Optimizer for
                non-Bayesian parameters (i.e. standard ``pytorch``
                parameters)
        '''
        self._parameters = parameters
        self._lrate = lrate
        self._std_optim = std_optim

    def zero_grad(self):
        'Set all the standard/Bayesian parameters gradient to zero.'
        if self._std_optim is not None:
            self._std_optim.zero_grad()
        for parameter in self._parameters:
            parameter.natural_grad.zero_()

    def step(self):
        'Update all the standard/Bayesian parameters.'
        if self._std_optim is not None:
            self._std_optim.step()
        for parameter in self._parameters:
            parameter.natural_grad_update(self._lrate)


class BayesianModelCoordinateAscentOptimizer(BayesianModelOptimizer):
    '''Optimizer that update iteratively groups of parameters. This
    optimizer is suited for model like PPCA which cannot estimate the
    gradient of all its paramaters at once.


    Example:
        >>> # Assume "model" is a BayesianModel to be trained and X is
        >>> # the dataset.
        >>> elbo_fn = beer.EvidenceLowerBound(len(X))
        >>> optim = beer.BayesianModelCoordinateAscentOptimizer(model.parameters)
        >>> for epoch in range(10):
        >>>     optim.zero_grad()
        >>>     elbo = elbo_fn(model, X)
        >>>     elbo.natural_backward()
        >>>     optim.step()

    '''

    def __init__(self, groups, lrate=1., std_optim=None):
        '''
        Args:
            ... (list): N List of ``BayesianParameter``.
                to be updated separately.
            lrate (float): learning rate.
            std_optim (``torch.optim.Optimizer``): Optimizer for
                non-Bayesian parameters (i.e. standard ``pytorch``
                parameters)
        '''
        parameters = []
        for group in groups:
            parameters += [param for param in group]
        super().__init__(parameters, lrate=lrate, std_optim=std_optim)
        self._groups = groups
        self._update_count = 0

    def step(self):
        'Update one group the standard/Bayesian parameters.'
        if self._std_optim is not None:
            self._std_optim.step()
        if self._update_count >= len(self._groups):
            self._update_count = 0
        for parameter in self._groups[self._update_count]:
            parameter.natural_grad_update(self._lrate)

        self._update_count += 1


__all__ = ['evidence_lower_bound', 'BayesianModelOptimizer',
           'BayesianModelCoordinateAscentOptimizer']
