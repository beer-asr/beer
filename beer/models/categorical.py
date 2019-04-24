import math
import torch

from .basemodel import Model
from .parameters import ConjugateBayesianParameter
from ..dists import Dirichlet


__all__ = ['Categorical', 'SBCategorical']


########################################################################
# Helper to build the default parameters.

def _default_param(weights, prior_strength, tensorconf):
    prior = Dirichlet.from_std_parameters(weights * prior_strength)
    posterior = Dirichlet.from_std_parameters(weights * prior_strength)
    return ConjugateBayesianParameter(prior, posterior)


def _default_sb_param(truncation, prior_strength):
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
    # The following property is exposed only for plotting/debugging
    # purposes.

    @property
    def mean(self):
        return self.weights.value()

    ####################################################################

    def sufficient_statistics(self, data):
        return self.weights.likelihood_fn.sufficient_statistics(data)

    def mean_field_factorization(self):
        return [[self.weights]]

    def expected_log_likelihood(self, stats):
        nparams = self.weights.natural_form()
        return self.weights.likelihood_fn(nparams, stats)

    def accumulate(self, stats):
        return {self.weights: stats.sum(dim=0)}


class SBCategorical(Model):
    'Categorical with a truncated stick breaking prior.'

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
        return cls(_default_sb_param(truncation, prior_strength))

    def __init__(self, stickbreaking):
        super().__init__()
        self.stickbreaking = stickbreaking
        device = self.stickbreaking.posterior.params.concentrations.device
        self.ordering = torch.arange(stickbreaking.posterior.dim[0], 
                                     device=device)
        #self.stickbreaking.register_callback(self._on_sticbreaking_update)

    def _on_sticbreaking_update(self):
        counts = self.stickbreaking.posterior.params.concentrations[:, 0]
        self.ordering = counts.sort(descending=True)[1]

    @property
    def reverse_ordering(self):
        reverse_ordering = torch.zeros_like(self.ordering)
        for i, j in enumerate(self.ordering):
            reverse_ordering[j] = i
        return reverse_ordering

    @property
    def mean(self):
        c = self.stickbreaking.posterior.params.concentrations
        s_dig =  torch.digamma(c.sum(dim=-1))
        log_v = torch.digamma(c[:, 0]) - s_dig
        log_1_v = torch.digamma(c[:, 1]) - s_dig
        log_prob = log_v
        log_prob[1:] += log_1_v[:-1].cumsum(dim=0)
        return log_prob.exp()[self.reverse_ordering]

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
        return stats[:, self.ordering] @ log_prob

    def accumulate(self, stats, parent_msg=None):
        self.ordering = stats.sum(dim=0).sort(descending=True)[1]
        ordered_stats = stats[:, self.ordering]
        s2 = ordered_stats.clone()
        s2 = torch.zeros_like(ordered_stats)
        s2[:, :-1] = ordered_stats[:, 1:]
        s2 = torch.flip(torch.flip(s2, dims=(1,)).cumsum(dim=1), dims=(1,))
        new_stats = torch.cat([ordered_stats[:, :, None], s2[:, :, None]], 
                              dim=-1)
        shape = new_stats.shape
        new_stats = new_stats.reshape(-1, 2)
        new_stats[:, -1] += new_stats[:, :-1].sum(dim=-1)
        new_stats = new_stats.reshape(*shape)
        return {self.stickbreaking: new_stats.sum(dim=0)}
