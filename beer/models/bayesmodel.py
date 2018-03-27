
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
        elif isinstance(value, BayesianModel):
            self._parameters += value.parameters
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
    def accumulate(self, s_stats, parent_message=None):
        '''Accumulate the sufficient statistics for the parameters's
        update.

        Args:
            s_stats (list): List of sufficient statistics.
            parent_message (object): Message from the parent (and
                the co-parents) to make the VB update.

        Returns:
            list: List of accumulated statistics for each parameter of
                the model. The list should be in the same order as the
                parameters were registered.

        '''
        NotImplemented

