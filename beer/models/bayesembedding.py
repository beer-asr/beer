
'''Bayesian Embedding Model.'''

import abc
import math
import torch
from torch import nn
from torch.autograd import Variable

from .bayesmodel import BayesianModel
from .mixture import _expand_labels


class BayesianEmbeddingModel(BayesianModel):
    'Bayesian Embedding Model (BEM).'

    def __init__(self, encoder, bayesian_model, nsamples=10):
        '''Initialize the BEM.

        Args:
            encoder (``MLPModel``): Encoder neural network..
            bayesian_model (``BayesianModel``): Bayesian Model of the
                embedding space.l

        '''
        super().__init__()
        self.encoder = encoder
        self.bayesian_model = bayesian_model
        self.nsamples = nsamples

    def evaluate(self, X, labels):
        'Convenience function mostly for plotting and debugging.'
        state = self.encoder(X)
        return state.mean, state.var

    def sufficient_statistics(self, X):
        self._state = self.encoder(X)
        nsamples = self.nsamples
        samples = []
        for i in range(self.nsamples):
            samples.append(self._state.sample())
        samples = torch.stack(samples).view(self.nsamples * X.size(0), -1)
        T = self.bayesian_model.sufficient_statistics(samples), \
            self.bayesian_model.sufficient_statistics_from_mean_var(
                self._state.mean, self._state.var)
        return T

    def forward(self, T, labels):
        T_s, T = T

        # log p(Z = labels | X)
        log_pred = self.bayesian_model.log_predictions(T_s).view(self.nsamples, T.size(0), -1)
        log_pred = log_pred.mean(dim=0)
        onehot_labels = _expand_labels(labels,
            len(self.bayesian_model.components)).type(T.type())
        log_p_labels = (onehot_labels * log_pred).sum(dim=-1)

        # Per-frame Max. Entropy distribution.
        preds = torch.exp(self.bayesian_model.log_predictions(T))
        nparams = self.bayesian_model.components._expected_nparams_as_matrix().data
        nparams = Variable(onehot_labels @ nparams)

        retval = log_p_labels - self._state.kl_div(nparams)
        self._labels = onehot_labels
        return retval

    def accumulate(self, T, parent_msg=None):
        return self.bayesian_model.accumulate(T[1].data, self._labels)

