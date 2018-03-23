
'Abstract Base Class for all "standard" Bayesian models.'

import abc
import torch

from ..expfamilyprior import kl_div


class BayesianParameter:
    '''Bayesian parameter is a parameter defined by a prior and a
    posterior distribution.

    Attributes:
        prior (``ExpFamilyPrior``): Prior distribution over the
            parameter.
        posterior (``ExpFamilyPrior``): Posterior distribution over
            the paramater.

    '''

    def __init__(self, prior, posterior):
        self.prior, self.posterior = prior, posterior
        self.natural_grad = torch.zeros_like(self.prior.natural_params)

    @property
    def expected_value(self):
        '''Expected value of the parameter w.r.t. the posterior
        distribution.

        '''
        return self.posterior.expected_sufficient_statistics

    def zero_natural_grad(self):
        self.natural_grad.zero_()


class BayesianParameterSet:
    '''Set of Bayesian parameters.

    The purpose of this class is only to register several parameters
    at once.

    Attributes:
        parameters (list): List of ``BayesianParameter``.

    '''

    def __init__(self, parameters):
        self._parameters = parameters

    def __len__(self):
        return len(self._parameters)

    def __getitem__(self, key):
        return self._parameters[key]


def kl_div_posterior_prior(parameters):
    '''Kullback-Leibler divergence between the posterior and the prior
    distribution of the paramater.

    Args:
        parameters (list): List of ``BayesianParameter``.

    Returns:
        float: KL( q || p)

    '''
    retval = 0.
    for parameter in parameters:
        retval += kl_div(parameter.posterior, parameter.prior)
    return retval


class BayesianModel(metaclass=abc.ABCMeta):
    'Abstract base class for Bayesian models.'

    def __init__(self):
        self._parameters = []

    def __setattr__(self, name, value):
        if isinstance(value, BayesianParameter):
            self._parameters.append(value)
        elif isinstance(value, BayesianParameterSet):
            for parameter in value:
                self._parameters.append(parameter)
        super().__setattr__(name, value)

    def __call__(self, X, labels=None):
        return self.forward(X, labels)

    @property
    def parameters(self):
        return self._parameters

    @abc.abstractmethod
    def forward(self, T, labels=None):
        '''Expected value of the log-likelihood w.r.t to the posterior
        distribution over the parameters.

        Args:
            T (Tensor[n_frames, dim]): Sufficient statistics.
            labels (LongTensor[n_frames]): Labels.

        Returns:
            Tensor: Per-frame expected value of the log-likelihood.

        '''
        NotImplemented

    @abc.abstractmethod
    def accumulate(self, s_stats, parent_message):
        '''Accumulate the sufficient statistics for the parameters's
        update.

        Args:
            s_stats (list): List of sufficient statistics.
            parent_message (object): Message from the parent (and
                the co-parents) to make the VB update.

        '''
        NotImplemented


class VariationalBayesLossInstance:
    '''Generic loss term.

    Note:
        This object should not be created directly.

    '''

    def __init__(self, expected_llh, kl_div, parameters, acc_stats, scale):
        self._exp_llh_per_frame = expected_llh
        self._kl_div = kl_div
        self._loss_per_frame = expected_llh - kl_div
        self._exp_llh = self._exp_llh_per_frame.sum()
        self._loss = self._loss_per_frame.sum()
        self._parameters = parameters
        self._acc_stats = acc_stats
        self._scale = scale

    def __imul__(self, other):
        self._loss *= other

    def __str__(self):
        return self._loss

    @property
    def value(self):
        'Value of the loss.'
        return self._loss

    @property
    def value_per_frame(sefl):
        'Per-frame value of the loss.'
        return self._loss_per_frame

    @property
    def exp_llh(self):
        'Expected log-likelihood.'
        return self._exp_llh

    @property
    def exp_llh_per_frame(sefl):
        'Per-frame expected log-likelihood.'
        return self._exp_llh_per_frame

    @property
    def kl_div(self):
        'Kullback-Leibler divergence.'
        return self._kl_div

    def backward_natural_grad(self):
        '''Accumulate the the natural gradient of the loss for each
        parameter of the model.

        '''
        for acc_stat, parameter in zip(acc_stats, self._parameters):
            parameter.natural_grad += parameter.prior.natural_params +  \
                scale * acc_stats - parameter.posterior.natural_params


class StochasticVariationalBayesLoss:
    '''Standard Variational Bayes Loss function.

    \ln p(x) \ge \langle \ln p(X|Z) \rangle_{q(Z)} - D_{kl}( q(Z) || p(Z))

    '''

    def __init__(self, model, datasize):
        '''Initialize the loss

        Args:
            model (``BayesianModel``): The model to use to compute the
                loss.
            datasize (int): Number of data point in the data set.

        '''
        self.model = model
        self.datasize = datasize

    def __call__(self, X):
        T = self.model.sufficient_statistics(X)
        return VariationalBayesLossInstance(
            expected_llh=self.model(T),
            kl_div=kl_div_prior(self.model),
            parameters=self.model.parameters,
            acc_stats=self.model.accumulate(T, parent_message=None),
            scale=float(len(X)) / self.datasize
        )

