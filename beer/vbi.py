
'Variational Bayes Inference.'


import torch.autograd as ta
from .models import BayesianModel

class EvidenceLowerBoundInstance:
    '''Evidence Lower Bound of a data set given a model.

    Note:
        This object should not be created directly.

    '''

    def __init__(self, expected_llh, kl_div, parameters, acc_stats, scale):
        self._elbo = scale * expected_llh.sum() - kl_div
        self._exp_llh_per_frame = expected_llh
        self._kl_div = kl_div
        self._parameters = parameters
        self._acc_stats = acc_stats
        self._scale = scale

    def __str__(self):
        return str(self._elbo)

    def __float__(self):
        return float(self._elbo)

    def scale(self, scale):
        self._elbo *= scale

    def backward(self):
        '''Compute the gradient of the loss w.r.t. to standard
        ``pytorch`` parameters.
        '''
        # Pytorch minimizes the loss ! We change the sign of the ELBO
        # just before to compute the gradient.
        (-self._elbo).backward()

    def natural_backward(self):
        '''Compute the natural gradient of the loss w.r.t. to all the
        :any:`BayesianParameter`.
        '''
        for parameter in self._parameters:
            try:
                acc_stats = self._acc_stats[parameter]
                parameter.natural_grad += parameter.prior.natural_hparams +  \
                    self._scale * acc_stats - \
                    parameter.posterior.natural_hparams
            except KeyError:
                pass


class EvidenceLowerBound:
    '''Evidence Lower Bound function.

    Args:
        model (:any:`BayesianModel`): The Bayesian model with which to
            compute the ELBO.
        data (``torch.Tensor``): The data set on which to evaluate the
            ELBO.
        latent_variables (object): Provide latent_variables to the model
            when computing the ELBO.

    Returns:
        ``EvidenceLowerBoundInstance``


    Example:
        >>> # Assume X is our data set and "model" is the model to be
        >>> # trained.
        >>> elbo_fn = beer.EvidenceLowerBound(len(X))
        >>> elbo = elbo_fn(model, X)
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

    def __init__(self, datasize):
        self.datasize = datasize

    def __call__(self, model, data, latent_variables=None):
        s_stats = model.sufficient_statistics(data)
        return EvidenceLowerBoundInstance(
            expected_llh=model(s_stats, latent_variables),
            kl_div=BayesianModel.kl_div_posterior_prior(model.parameters),
            parameters=model.parameters,
            acc_stats=model.accumulate(s_stats),
            scale=float(len(data)) / self.datasize
        )


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
            parameter.zero_natural_grad()

    def step(self):
        'Update all the standard/Bayesian parameters.'
        if self._std_optim is not None:
            self._std_optim.step()
        for parameter in self._parameters:
            parameter.posterior.natural_hparams = ta.Variable(
                parameter.posterior.natural_hparams + \
                self._lrate * parameter.natural_grad,
                requires_grad=True
            )

