import math
import torch

from .modelset import ModelSet
from .parameters import ConjugateBayesianParameter
from .categorical import Categorical, _default_param


__all__ = ['CategoricalSet']


class CategoricalSet(ModelSet):
    'Set of Categorical distributions with a Dirichlet prior.'

    @classmethod
    def create(cls, weights, prior_strength=1.):
        '''Create a Categorical model.

        Args:
            weights (``torch.Tensor[dim]``): Initial mean distribution.
            prior_strength (float): Strength of the Dirichlet prior.

        Returns:
            :any:`Categorical`

        '''
        return cls(_default_param(weights.detach(), prior_strength))

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    ####################################################################

    def sufficient_statistics(self, data):
        return self.weights.likelihood_fn.sufficient_statistics(data)

    def mean_field_factorization(self):
        return [[self.weights]]

    def expected_log_likelihood(self, stats):
        nparams = self.weights.natural_form()
        return self.weights.likelihood_fn(nparams, stats)

    def accumulate(self, stats, resps):
        w_stats = resps.t() @ stats
        return {self.weights: w_stats}

    def accumulate_from_jointresps(self, jointresps_stats):
        return {self.weights: jointresps_stats.sum(dim=0)}

    ####################################################################
    # ModelSet interface.

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self.weights[key])
        return Categorical(self.weights[key])

