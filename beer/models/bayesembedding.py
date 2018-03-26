
'''Bayesian Embedding Model.'''

import abc
import math
import torch
from torch import nn
from torch.autograd import Variable

from .bayesmodel import BayesianModel


class BayesianEmbeddingModel(BayesianModel):
    'Bayesian Embedding Model (BEM).'

    def __init__(self, encoder, bayesian_model):
        '''Initialize the BEM.

        Args:
            encoder (``MLPModel``): Encoder neural network..
            bayesian_model (``BayesianModel``): Bayesian Model of the
                embedding space.l

        '''
        super().__init__()
        self.encoder = encoder
        self.bayesian_model = bayesian_model

    def evaluate(self, data, labels):
        'Convenience function mostly for plotting and debugging.'
        state = self(data, sampling=False)
        loss, llh, kld = self.loss(data, labels, state)
        return -loss, llh, kld, state['encoder_state'].mean, \
            state['encoder_state'].std_dev ** 2

    def sufficient_statistics(self, X):
        self._state = self.encoder(X)
        return self.bayesian_model.sufficient_statistics_from_mean_var(
            self._state.mean, self._state.var)

    def forward(self, T, labels=None):
        retval = self.bayesian_model(T, labels) + self._state.entropy()
        self._state = None
        return retval

    def accumulate(self, T, parent_msg=None):
        return self.bayesian_model.accumulate(T, parent_message)

