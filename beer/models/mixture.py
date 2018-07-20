
'Bayesian Mixture model.'

import torch
from .bayesmodel import DiscreteLatentBayesianModel
from .bayesmodel import BayesianParameter
from ..expfamilyprior import DirichletPrior
from ..utils import onehot
from ..utils import logsumexp


class Mixture(DiscreteLatentBayesianModel):
    '''Bayesian Mixture Model.'''

    @classmethod
    def create(cls, weights, prior_strength, modelset):
        prior_weights = DirichletPrior(prior_strength * weights)
        posterior_weights = DirichletPrior(prior_strength * weights)
        return cls(prior_weights, posterior_weights, modelset)

    def __init__(self, prior_weights, posterior_weights, modelset):
        '''
        Args:
            prior_weights (:any:`DirichletPrior`): Prior distribution
                over the weights of the mixture.
            posterior_weights (any:`DirichletPrior`): Posterior
                distribution over the weights of the mixture.
            modelset (:any:`BayesianModelSet`): Set of models.

        '''
        super().__init__(modelset)
        self.weights_param = BayesianParameter(prior_weights, posterior_weights)

    @property
    def weights(self):
        'Expected value of the weights of the mixture.'
        weights = torch.exp(self.weights_param.expected_value())
        return weights / weights.sum()

    def _local_kl_divergence(self, log_resps):
        log_weights = self.weights_param.expected_value()
        retval = torch.sum(log_resps.exp() * (log_resps - log_weights), dim=-1)
        return retval

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def mean_field_factorization(self):
        mf_groups = self.modelset.mean_field_factorization()
        mf_groups[0].append(self.weights_param)
        return mf_groups

    def forward(self, s_stats, labels=None):
        log_weights = self.weights_param.expected_value().view(1, -1)
        per_component_exp_llh = self.modelset(s_stats)

        if labels is not None:
            resps = onehot(labels, len(self.modelset),
                           dtype=log_weights.dtype, device=log_weights.device)
            exp_llh = (per_component_exp_llh * resps).sum(dim=-1)
            self.cache['resps'] = resps
        else:
            w_per_component_exp_llh = per_component_exp_llh + log_weights
            exp_llh = logsumexp(w_per_component_exp_llh, dim=1).view(-1)
            log_resps = w_per_component_exp_llh - exp_llh.view(-1, 1).detach()
            local_kl_div = self._local_kl_divergence(log_resps)
            resps = log_resps.exp()
            exp_llh = (per_component_exp_llh * resps).sum(dim=-1)
            self.cache['resps'] = log_resps.exp()

        return exp_llh - local_kl_div

    def accumulate(self, s_stats, parent_msg=None):
        resps = self.cache['resps']
        retval = {
            self.weights_param: resps.sum(dim=0),
            **self.modelset.accumulate(s_stats, resps)
        }
        self.clear_cache()
        return retval

    def local_kl_div_posterior_prior(self, parent_msg=None):
        return self.modelset.local_kl_div_posterior_prior(self.cache['resps'])

    ####################################################################
    # DiscreteLatentBayesianModel interface.
    ####################################################################

    def posteriors(self, data):
        s_stats = self.modelset.sufficient_statistics(data)
        per_component_exp_llh = self.modelset(s_stats)
        per_component_exp_llh += self.weights_param.expected_value().view(1, -1)
        lognorm = logsumexp(per_component_exp_llh, dim=1).view(-1)
        return torch.exp(per_component_exp_llh - lognorm.view(-1, 1))


def create(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    prior_strength = model_conf['prior_strength']
    modelset = create_model_handle(model_conf['components'], mean, variance)
    size = len(modelset)
    weights = torch.ones(size, dtype=dtype, device=device) / size
    return Mixture.create(weights, prior_strength, modelset)


__all__ = ['Mixture']
