import math
import torch

from .basemodel import Model
from .modelset import ModelSet
from .parameters import ConjugateBayesianParameter
from .categorical import Categorical, _default_param
from ..dists import Dirichlet


__all__ = ['CategoricalSet', 'SBCategoricalSet']


########################################################################
# Helper to build the default parameters.

def _default_set_sb_param(n_components, root_sb_categorical, prior_strength):
    mean = root_sb_categorical.mean
    params = torch.ones(n_components, len(root_sb_categorical.stickbreaking), 2)
    params[:, :, 0] = prior_strength * mean 
    params[:, :, 1] = prior_strength * (1 - mean.cumsum(dim=0))
    params = params.reshape(-1, 2)
    prior = Dirichlet.from_std_parameters(params)
    params = root_sb_categorical.stickbreaking.posterior.params.concentrations.repeat(n_components, 1)
    posterior = Dirichlet.from_std_parameters(params.clone())
    return ConjugateBayesianParameter(prior, posterior)

########################################################################


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


class SBCategoricalSet(Model):
    'Set of categorical with a truncated stick breaking prior.'

    # NOTE: contrary to the single SBCategorical, this class does not
    # include a re-ordering mechanism.  

    @classmethod
    def create(cls, n_components, root_sb_categorical, prior_strength=1.):
        '''Create a set of Categorical model.
        Args:
            n_components (int): Number of components in the set.
            root_sb_categorical (int): Root stick-breaking process.
            prior_strength (float): Strength (i.e. concentration) of
                the bottom level stick breaking prior.
        Returns:
            :any:`SBCategoricalSet`
        '''
        return cls(n_components,
                   _default_set_sb_param(n_components, root_sb_categorical, 
                                         prior_strength))

    def __init__(self, n_components, stickbreaking):
        super().__init__()
        self.n_components = n_components
        self.stickbreaking = stickbreaking

    @property
    def mean(self):
        c = self.stickbreaking.posterior.params.concentrations
        c = c.reshape(self.n_components, -1, 2)
        s_dig = torch.digamma(c.sum(dim=-1))
        log_v = torch.digamma(c[:, :, 0]) - s_dig
        log_1_v = torch.digamma(c[:, :, 1]) - s_dig
        log_prob = log_v
        log_prob[:, 1:] += log_1_v[:, :-1].cumsum(dim=1)
        return log_prob.exp()

    ####################################################################

    def sufficient_statistics(self, data):
        # Data is a matrix of one-hot encoding vectors.
        return data

    def mean_field_factorization(self):
        return [[self.stickbreaking]]

    def expected_log_likelihood(self, stats):
        c = self.stickbreaking.posterior.params.concentrations
        c = c.reshape(self.n_components, -1, 2)
        s_dig = torch.digamma(c.sum(dim=-1))
        log_v = torch.digamma(c[:, :, 0]) - s_dig
        log_1_v = torch.digamma(c[:, :, 1]) - s_dig
        log_prob = log_v
        log_prob[:, 1:] += log_1_v[:, :-1].cumsum(dim=1)
        pad = torch.ones_like(log_1_v)
        self.cache['sb_stats'] = torch.cat([log_1_v[:, :, None],
                                            pad[:, :, None]], dim=-1)
        return log_prob @ stats

    def accumulate(self, stats):
        raise NotImplementedError 

    def accumulate_from_jointresps(self, stats):
        s2 = torch.zeros_like(stats)
        s2[:, :, :-1] = stats[:, :,  1:]
        s2 = torch.flip(torch.flip(s2, dims=(2,)).cumsum(dim=2), dims=(2,))
        new_stats = torch.cat([stats[:, :, :, None], s2[:, :, :, None]],
                               dim=-1).sum(dim=0)
        new_stats = new_stats.reshape(self.n_components, -1,  2)
        new_stats[:, :, -1] += new_stats[:, :, :-1].sum(dim=-1)
        return {self.stickbreaking: new_stats.reshape(-1, 2)}
