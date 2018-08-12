
'Bayesian Mixture model.'

import torch
from .bayesmodel import DiscreteLatentBayesianModel
from .bayesmodel import BayesianParameter
from ..priors import DirichletPrior
from ..utils import onehot
from ..utils import logsumexp


class Mixture(DiscreteLatentBayesianModel):
    '''Bayesian Mixture Model.'''

    @classmethod
    def create(cls, modelset, weights=None, prior_strength=1.):
        '''Create a mixture model.

        Args:
            modelset (:any:`BayesianModelSet`): Component of the
                mixture.
            weights (``torch.Tensor[k]``): Prior probabilities of
                the components of the mixture. If not provided, assume
                flat prior.
            prior_strength (float): Strength of the prior over the
                weights.

        '''
        prior_nparams = modelset.mean_field_groups[0][0].prior.natural_parameters
        dtype, device = prior_nparams.dtype, prior_nparams.device

        if weights is None:
            weights = torch.ones(len(modelset), dtype=dtype, device=device)
            weights /= len(modelset)
        else:
            weights = torch.tensor(weights, dtype=dtype, device=device,
                                   requires_grad=False)
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
        self.weights = BayesianParameter(prior_weights, posterior_weights)

    def _local_kl_divergence(self, log_resps):
        log_weights = self.weights.expected_natural_parameters()
        retval = torch.sum(log_resps.exp() * (log_resps - log_weights), dim=-1)
        return retval

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        mf_groups = self.modelset.mean_field_factorization()
        mf_groups[0].append(self.weights)
        return mf_groups

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, labels=None):
        # Per-components weighted log-likelihood.
        log_weights = self.weights.expected_natural_parameters().view(1, -1)
        per_component_exp_llh = self.modelset.expected_log_likelihood(stats)
        w_per_component_exp_llh = (per_component_exp_llh + log_weights)

        # Responsibilities and expected llh.
        exp_llh = logsumexp(w_per_component_exp_llh.detach(), dim=1).view(-1)
        log_resps = w_per_component_exp_llh.detach() - exp_llh.view(-1, 1)
        local_kl_div = self._local_kl_divergence(log_resps)
        resps = log_resps.exp()
        exp_llh = (w_per_component_exp_llh * resps).sum(dim=-1)

        # If some labels are provided, override the previous results.
        if labels is not None:
            idxs = labels > -1
            if idxs.sum() > 0:
                labels_resps = onehot(labels, len(self.modelset),
                            dtype=log_weights.dtype, device=log_weights.device)
                resps[idxs] = labels_resps[idxs]
                local_kl_div[idxs] = 0.
                exp_llh = (w_per_component_exp_llh * resps).sum(dim=-1)

        # Store the responsibilites to accumulate the statistics.
        self.cache['resps'] = resps

        return exp_llh - local_kl_div

    def accumulate(self, stats, parent_msg=None):
        resps = self.cache['resps']
        retval = {
            self.weights: resps.sum(dim=0),
            **self.modelset.accumulate(stats, resps)
        }
        return retval


    ####################################################################
    # DiscreteLatentBayesianModel interface.
    ####################################################################

    def posteriors(self, data):
        stats = self.modelset.sufficient_statistics(data)
        per_component_exp_llh = self.modelset.expected_log_likelihood(stats)
        per_component_exp_llh += self.weights.expected_value().view(1, -1)
        lognorm = logsumexp(per_component_exp_llh, dim=1).view(-1)
        return torch.exp(per_component_exp_llh - lognorm.view(-1, 1))


__all__ = ['Mixture']
