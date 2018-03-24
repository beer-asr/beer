
'Variational Bayes Inference.'


import numpy as np
import torch.autograd as ta

from .models import kl_div_posterior_prior

class VariationalBayesLossInstance:
    '''Generic loss term.

    Note:
        This object should not be created directly.

    '''

    def __init__(self, expected_llh, kl_div, parameters, acc_stats, scale):
        self._exp_llh = expected_llh.sum()
        self._loss = scale * self._exp_llh - kl_div
        self._exp_llh_per_frame = expected_llh
        self._kl_div = kl_div
        self._parameters = parameters
        self._acc_stats = acc_stats
        self._scale = scale

    def __imul__(self, other):
        self._loss *= other

    def __str__(self):
        return str(self._loss)

    def __float__(self):
        return float(self.value)

    @property
    def value(self):
        'Value of the loss.'
        return self._loss

    @property
    def exp_llh(self):
        'Expected log-likelihood.'
        return self._exp_llh

    @property
    def exp_llh_per_frame(self):
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
        for acc_stats, parameter in zip(self._acc_stats, self._parameters):
            parameter.natural_grad += parameter.prior.natural_params +  \
                self._scale * acc_stats - \
                parameter.posterior.natural_params

    def backward(self):
        # Pytorch minimizes the loss ! We change the sign of the loss
        # just before to compute the gradient.
        (-self._loss).backward()


class StochasticVariationalBayesLoss:
    '''Standard Variational Bayes Loss function.

    \ln p(x) \ge \langle \ln p(X|Z) \rangle_{q(Z)} - D_{kl}( q(Z) || p(Z))

    '''

    def __init__(self, datasize):
        '''Initialize the loss

        Args:
            model (``BayesianModel``): The model to use to compute the
                loss.
            datasize (int): Number of data point in the data set.

        '''
        self.datasize = datasize

    def __call__(self, model, X, labels=None):
        T = model.sufficient_statistics(X)
        return VariationalBayesLossInstance(
            expected_llh=model(T, labels),
            kl_div=kl_div_posterior_prior(model.parameters),
            parameters=model.parameters,
            acc_stats=model.accumulate(T),
            scale=float(len(X)) / self.datasize
        )


class BayesianModelOptimizer:
    'Bayesian Model Optimizer.'

    def __init__(self, parameters, lrate=1.):
        '''Initialize the optimizer.

        Args:
            parameters (list): List of ``BayesianParameters``.
            lrate (float): learning rate.

        '''
        self._parameters = parameters
        self._lrate = lrate

    def zero_natural_grad(self):
        for parameter in self._parameters:
            parameter.zero_natural_grad()

    def step(self):
        for parameter in self._parameters:
            parameter.posterior.natural_params = ta.Variable(
                parameter.posterior.natural_params + \
                self._lrate * parameter.natural_grad,
                requires_grad=True
            )

