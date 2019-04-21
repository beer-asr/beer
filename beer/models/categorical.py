import math
import torch

from .basemodel import Model
from .parameters import ConjugateBayesianParameter
from ..dists import Dirichlet


__all__ = ['Categorical']


########################################################################
# Helper to build the default parameters.

def _default_param(weights, prior_strength, tensorconf):
    prior = Dirichlet.from_std_parameters(weights * prior_strength)
    posterior = Dirichlet.from_std_parameters(weights * prior_strength)
    return ConjugateBayesianParameter(prior, posterior)


def _default_sb_param(weights, truncation, prior_strength, tensorconf):
    params = torch.ones(truncation, 2)
    params[:, 1] = prior_strength
    prior = Dirichlet.from_std_parameters(params)
    posterior = Dirichlet.from_std_parameters(params.clone())
    return ConjugateBayesianParameter(prior, posterior)

########################################################################

class Categorical(Model):
    'Categorical distribution with a Dirichlet prior.'

    @classmethod
    def create(cls, weights, prior_strength=1.):
        '''Create a Categorical model.

        Args:
            weights (``torch.Tensor[dim]``): Initial mean distribution.
            prior_strength (float): Strength of the Dirichlet prior.

        Returns:
            :any:`Categorical`

        '''
        tensorconf = {'dtype': weights.dtype, 'device': weights.device,
                      'requires_grad': False}
        return cls(_default_param(weights.detach(), prior_strength, tensorconf))

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

    def accumulate(self, stats, parent_msg=None):
        return {self.weights: stats.sum(dim=0)}



class SBCategorical(Model):
    'Categorical with a stick breaking prior.'

    @classmethod
    def create(cls, truncation, prior_strength=1.):
        '''Create a Categorical model.

        Args:
            truncation (int): Truncation of the stick breaking process.
            prior_strength (float): Strength (i.e. concentration) of
                the stick breaking prior.

        Returns:
            :any:`SBCategorical`

        '''
        return cls(_default_sb_param(prior_strength))

    def __init__(self, stickbreaking):
        super().__init__()
        self.stickbreaking= stickbreaking

    ####################################################################

    def sufficient_statistics(self, data):
        # Data is a matrix of one-hot encoding vectors.
        return data

    def mean_field_factorization(self):
        return [[self.stickbreaking]]

    def expected_log_likelihood(self, stats):
        c = self.stickbreaking.posterior.params.concentrations
        s_dig =  torch.digamma(c.sum(dim=-1))
        log_v = torch.digamma(c[:, 0]) - s_dig
        log_1_v = torch.digamma(c[:, 1]) - s_dig
        log_prob = log_v
        log_prob[1:] += log_1_v[:-1].cumsum(dim=0)
        return log_prob

    def accumulate(self, stats, parent_msg=None):
        return {self.stickbreaking: stats.sum(dim=0)}

