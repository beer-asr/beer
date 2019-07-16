import math
import torch

from .basemodel import Model
from .parameters import ConjugateBayesianParameter
from ..dists import Dirichlet
from ..dists import Gamma


__all__ = ['Categorical', 'SBCategorical', 'SBCategoricalHyperPrior']


########################################################################
# Helper to build the default parameters.

def _default_param(weights, prior_strength):
    prior = Dirichlet.from_std_parameters(weights * prior_strength)
    posterior = Dirichlet.from_std_parameters(weights * prior_strength)
    return ConjugateBayesianParameter(prior, posterior)


def _default_sb_param(truncation, prior_strength):
    params = torch.ones(truncation, 2)
    params[:, 1] = prior_strength
    prior = Dirichlet.from_std_parameters(params)
    posterior = Dirichlet.from_std_parameters(params.clone())
    return ConjugateBayesianParameter(prior, posterior)


def _default_concentration_param(mean, prior_strength):
    shape = torch.ones_like(mean) * prior_strength
    rate = prior_strength / mean
    prior = Gamma.from_std_parameters(shape, rate)
    posterior = Gamma.from_std_parameters(shape.clone(), rate.clone())
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
        return cls(_default_param(weights.detach(), prior_strength))

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    @property
    def mean(self):
        return self.weights.value()

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
        self.stickbreaking.register_callback(self._transform_stats, 
                                             notify_before_update=True)

    def _transform_stats(self):
        stats = self.stickbreaking.stats
        self.ordering = stats.sort(descending=True)[1]
        stats = stats[self.ordering]
        s2 = torch.zeros_like(stats)
        s2[:-1] = stats[1:]
        s2 = torch.flip(torch.flip(s2, dims=(0,)).cumsum(dim=0), dims=(0,))
        new_stats = torch.cat([stats[:, None], s2[:, None]], dim=-1)
        new_stats[:, -1] += new_stats[:, :-1].sum(dim=-1)
        self.stickbreaking.stats = new_stats[self.reverse_ordering, :]

    def _log_v(self):
        c = self.stickbreaking.posterior.params.concentrations[self.ordering]
        s_dig = torch.digamma(c.sum(dim=-1))
        log_v = torch.digamma(c[:, 0]) - s_dig
        log_1_v = torch.digamma(c[:, 1]) - s_dig
        return log_v, log_1_v

    def _log_prob(self):
        log_v, log_1_v = self._log_v()
        log_prob = log_v
        log_prob[1:] += log_1_v[:-1].cumsum(dim=0)
        return log_prob, log_1_v
        
    @property
    def reverse_ordering(self):
        reverse_ordering = torch.zeros_like(self.ordering)
        for i, j in enumerate(self.ordering):
            reverse_ordering[j] = i
        return reverse_ordering

    @property
    def mean(self):
        c = self.stickbreaking.posterior.params.concentrations[self.ordering]
        norm = c.sum(dim=-1) + torch.finfo(c.dtype).eps
        weights = c[:, 0] / norm
        residual = (c[:, 1] / norm).cumprod(dim=0)
        weights[1:] *= residual[:-1]
        return weights[self.reverse_ordering]

    ####################################################################

    def sufficient_statistics(self, data):
        # Data is a matrix of one-hot encoding vectors.
        return data

    def mean_field_factorization(self):
        return [[self.stickbreaking]]

    def expected_log_likelihood(self, stats):
        log_prob, _ = self._log_prob()
        return stats @ log_prob[self.reverse_ordering]

    def accumulate(self, stats):
        return {self.stickbreaking: stats.sum(dim=0)}


class SBCategoricalHyperPrior(SBCategorical):
    '''Categorical with a truncated stick breaking prior and a hyper-prior
    over the concentration parameter of the stick breaking process.

    '''

    @classmethod
    def create(cls, truncation, prior_strength=1., hyper_prior_strength=1.):
        '''Create a Categorical model.

        Args:
            truncation (int): Truncation of the stick breaking process.
            prior_strength (float): Strength (i.e. concentration) of
                the stick breaking prior.
            hyper_rior_strength (float): Strength of hyper-prior over the
                concentration parameter of the stick-breaking process.

        Returns:
            :any:`SBCategoricalHyperPrior`

        '''
        concentration = _default_concentration_param(
            torch.ones(1) * prior_strength,
            hyper_prior_strength)
        sb = _default_sb_param(truncation, prior_strength)
        return cls(sb, concentration)

    def __init__(self, stickbreaking, concentration):
        super().__init__(stickbreaking)
        self.concentration = concentration
        self.stickbreaking.register_callback(self._on_stickbreaking_update)
        self.concentration.register_callback(self._on_concentration_update)
        self._on_concentration_update()

    def _on_concentration_update(self):
        self.stickbreaking.prior.params.concentrations[:, 1] = \
                self.concentration.value()

    def _on_stickbreaking_update(self):
        _, log_1_v = self._log_prob()
        pad = torch.ones_like(log_1_v)
        sb_stats = torch.cat([log_1_v[:, None], pad[:, None]], dim=-1)
        self.concentration.stats = sb_stats.sum(dim=0)
        self.concentration.natural_grad_update(lrate=1.)
