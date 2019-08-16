'Bayesian Mixture model.'

from operator import mul
import torch
from .basemodel import DiscreteLatentModel
from .parameters import ConjugateBayesianParameter
from .categorical import Categorical
from ..utils import onehot


__all__ = ['Mixture']


class Mixture(DiscreteLatentModel):
    '''Bayesian Mixture Model.'''

    @classmethod
    def create(cls, modelset, categorical=None, prior_strength=1.):
        '''Create a mixture model.

        Args:
            modelset (:any:`BayesianModelSet`): Component of the
                mixture.
            categorical (``Categorical``): Categorical model of the
                mixing weights.
            prior_strength (float): Strength of the prior over the
                weights.
        '''
        mf_groups = modelset.mean_field_factorization()
        tensor = mf_groups[0][0].prior.natural_parameters()
        tensorconf = {'dtype': tensor.dtype, 'device': tensor.device,
                      'requires_grad': False}

        if categorical is None:
            weights = torch.ones(len(modelset), **tensorconf)
            weights /= len(modelset)
            categorical = Categorical.create(weights, prior_strength)
        return cls(categorical, modelset)

    def __init__(self, categorical, modelset):
        super().__init__(modelset)
        self.categorical = categorical

    # Log probability of each components.
    def _log_weights(self, tensorconf):
        data = torch.eye(len(self.modelset), **tensorconf)
        stats = self.categorical.sufficient_statistics(data)
        return self.categorical.expected_log_likelihood(stats)

    def _local_kl_divergence(self, log_resps, log_weights):
        retval = torch.sum(log_resps.exp() * (log_resps - log_weights), dim=-1)
        return retval

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        l1 = self.modelset.mean_field_factorization()
        l2 = self.categorical.mean_field_factorization()
        diff = len(l1) - len(l2)
        if diff > 0:
            l2 += [[] for _ in range(abs(diff))]
        else:
            l1 += [[] for _ in range(abs(diff))]
        return [u + v for u, v in zip(l1, l2)]

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, labels=None, **kwargs):
        # Per-components weighted log-likelihood.
        tensorconf = {'dtype': stats.dtype, 'device': stats.device}
        log_weights = self._log_weights(tensorconf)[None]
        per_component_exp_llh = self.modelset.expected_log_likelihood(stats,
                                                                      **kwargs)

        # Responsibilities and expected llh.
        if labels is None:
            w_per_component_exp_llh = (per_component_exp_llh + log_weights).detach()
            lnorm = torch.logsumexp(w_per_component_exp_llh, dim=1).view(-1, 1)
            log_resps = w_per_component_exp_llh - lnorm
            resps = log_resps.exp()
            local_kl_div = self._local_kl_divergence(log_resps, log_weights)
        else:
            resps = onehot(labels, len(self.modelset),
                            dtype=log_weights.dtype, device=log_weights.device)
            local_kl_div = 0.

        # Store the responsibilites to accumulate the statistics.
        self.cache['resps'] = resps

        exp_llh = (per_component_exp_llh * resps).sum(dim=-1)
        return exp_llh - local_kl_div

    def accumulate(self, stats):
        resps = self.cache['resps']
        resps_stats = self.categorical.sufficient_statistics(resps)
        retval = {
            **self.categorical.accumulate(resps_stats),
            **self.modelset.accumulate(stats, resps)
        }
        return retval


    ####################################################################
    # DiscreteLatentModel interface.
    ####################################################################

    def posteriors(self, data):
        stats = self.modelset.sufficient_statistics(data)
        
        # Per-components weighted log-likelihood.
        tensorconf = {'dtype': stats.dtype, 'device': stats.device}
        log_weights = self._log_weights(tensorconf)[None]
        per_component_exp_llh = self.modelset.expected_log_likelihood(stats)
        
        lognorm = torch.logsumexp(per_component_exp_llh, dim=1).view(-1)
        return torch.exp(per_component_exp_llh - lognorm.view(-1, 1))
