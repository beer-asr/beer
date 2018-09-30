'Objective functions.'


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
    __repr_str = '{classname}(value={value})'

    def __init__(self, elbo_value, acc_stats, model_parameters, minibatchsize,
                 datasize):
        self._elbo_value = elbo_value
        self._acc_stats = acc_stats
        self._model_parameters = set(model_parameters)
        self._minibatchsize = minibatchsize
        self._datasize = datasize

    def __repr__(self):
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            value=float(self._elbo_value)
        )

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
        # Pytorch minimizes the loss ! We change the sign of the ELBO
        # just before to compute the gradient.
        if self._elbo_value.requires_grad:
            (-self._elbo_value).backward()

        scale = self._datasize / self._minibatchsize
        for parameter in self._model_parameters:
            acc_stats = self._acc_stats[parameter]
            parameter.store_stats(scale * acc_stats)


class CollapsedEvidenceLowerBoundInstance:
    '''Collapsed Evidence Lower Bound of a data set given a model.

    Note:
        This object should not be created directly.

    '''
    __repr_str = '{classname}(value={value})'

    def __init__(self, elbo_value, acc_stats, model_parameters):
        self._elbo_value = elbo_value
        self._acc_stats = acc_stats
        self._model_parameters = set(model_parameters)

    def __repr__(self):
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            value=float(self._elbo_value)
        )

    def __str__(self):
        return str(self._elbo_value)

    def __float__(self):
        return float(self._elbo_value)

    def backward(self):
        # Pytorch minimizes the loss ! We change the sign of the ELBO
        # just before to compute the gradient.
        if self._elbo_value.requires_grad:
            (-self._elbo_value).backward()
            
        for parameter in self._model_parameters:
            acc_stats = self._acc_stats[parameter].detach()
            parameter.store_stats(acc_stats)

        return self._acc_stats
    
class StochasticCollapsedEvidenceLowerBoundInstance:
    '''Collapsed Evidence Lower Bound of a data set given a model.

    Note:
        This object should not be created directly.

    '''
    __repr_str = '{classname}(value={value})'

    def __init__(self, elbo_value, acc_stats, model_parameters, minibatchsize,
                 datasize):
        self._elbo_value = elbo_value
        self._acc_stats = acc_stats
        self._model_parameters = set(model_parameters)
        self._minibatchsize = minibatchsize
        self._datasize = datasize

    def __repr__(self):
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            value=float(self._elbo_value)
        )

    def __str__(self):
        return str(self._elbo_value)

    def __float__(self):
        return float(self._elbo_value)

    def backward(self):
        # Pytorch minimizes the loss ! We change the sign of the ELBO
        # just before to compute the gradient.
        if self._elbo_value.requires_grad:
            (-self._elbo_value).backward()
            
        scale = self._datasize / self._minibatchsize
        for parameter in self._model_parameters:
            acc_stats = self._acc_stats[parameter].detach()
            parameter.store_stats(scale * acc_stats)

        return self._acc_stats
    

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
    scale = datasize / float(mb_datasize)
    stats = model.sufficient_statistics(minibatch_data)
    exp_llh = model.expected_log_likelihood(stats, **kwargs)
    if not fast_eval:
        kl_div = model.kl_div_posterior_prior().sum()
    else:
        kl_div = 0.
    elbo_value = float(scale) * exp_llh.sum() - kl_div
    acc_stats = model.accumulate(torch.tensor(stats))
    model.clear_cache()

    return EvidenceLowerBoundInstance(elbo_value, acc_stats,
                                      model.bayesian_parameters(),
                                      mb_datasize, datasize)


def collapsed_evidence_lower_bound(model=None, minibatch_data=None, **kwargs):
    '''Collapsed Evidence Lower Bound objective function of Variational
    Bayes Inference.

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
        ``CollapsedEvidenceLowerBoundInstance``

    '''
    stats = model.sufficient_statistics(minibatch_data)
    elbo_value = model.marginal_log_likelihood(stats, **kwargs).sum()
    acc_stats = model.accumulate(torch.tensor(stats))
    model.clear_cache()
    return CollapsedEvidenceLowerBoundInstance(elbo_value, acc_stats,
                                               model.bayesian_parameters())


def stochastic_collapsed_evidence_lower_bound(model, minibatch_data, datasize=-1, 
                                              **kwargs):
    '''Collapsed Evidence Lower Bound objective function of Variational
    Bayes Inference.

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
        ``CollapsedEvidenceLowerBoundInstance``

    '''
    if model is None and  minibatch_data is None and datasize > 0:
        return EvidenceLowerBoundInstance(0., {}, [], 0, datasize)
    elif model is None or minibatch_data is None:
        raise ValueError('if datasize is not provided, need at least "model" '
                         'and "minibatch_data"')

    mb_datasize = len(minibatch_data)
    if datasize <= 0:
        datasize = mb_datasize
    scale = datasize / float(mb_datasize)
    stats = model.sufficient_statistics(minibatch_data)
    exp_llh = model.marginal_log_likelihood(stats, **kwargs)
    kl_div = model.kl_div_posterior_prior().sum()
    elbo_value = float(scale) * exp_llh.sum() - kl_div
    acc_stats = model.accumulate(torch.tensor(stats))
    model.clear_cache()

    return EvidenceLowerBoundInstance(elbo_value, acc_stats,
                                      model.bayesian_parameters(),
                                      mb_datasize, datasize)




__all__ = ['evidence_lower_bound', 'collapsed_evidence_lower_bound',
           'stochastic_collapsed_evidence_lower_bound']

