
'Bayesian Language Models.'

import abc
import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel
from ..priors import DirichletPrior
from ..utils import onehot


class UnigramLM(BayesianModel):
    'Bayesian unigram LM.'

    @classmethod
    def create(cls, voc_size, prior_strength=1.):
        '''Create a Normal model.

        Args:
            voc_size (int): Size of the vocabulary.

        Returns:
            :any:`UnigramLM`

        '''
        weights = torch.ones(voc_size) / voc_size
        prior_weights = DirichletPrior(weights * prior_strength)
        posterior_weights = DirichletPrior(weights * prior_strength)
        return cls(prior_weights, posterior_weights)

    def __init__(self, prior, posterior):
        super().__init__()
        self.weights = BayesianParameter(prior, posterior)

    @property
    def voc_size(self):
        'Size of the vocabulary.'
        return len(self.weights.expected_value())

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def sufficient_statistics(self, data):
        return onehot(data, self.voc_size, data.dtype, data.device)

    def mean_field_factorization(self):
        return [[self.weights]]

    def expected_log_likelihood(self, stats):
        log_weight = self.weights.expected_natural_parameters()
        return log_weight[stats]

    def accumulate(self, stats):
        dtype = self.weights.expected_natural_parameters().dtype
        return {self.weights: stats.sum(dim=0).type(dtype)}


__all__ = [
    'UnigramLM'
]
