'Bayesian Mixture model.'

from operator import mul
import torch
from .basemodel import DiscreteLatentModel
from .parameters import ConjugateBayesianParameter
from ..dists import Dirichlet
from ..dists import DirichletStdParams
from ..utils import onehot


__all__ = ['Mixture']


########################################################################
# Helper to build the default parameters.

def _default_param(weights, prior_strength):
    params = DirichletStdParams(prior_strength * weights)
    prior_weights = Dirichlet(params)
    params = DirichletStdParams(prior_strength * weights)
    posterior_weights = Dirichlet(params)
    return ConjugateBayesianParameter(prior_weights, posterior_weights)

########################################################################


class Mixture(DiscreteLatentModel):
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
        mf_groups = modelset.mean_field_factorization()
        tensor = mf_groups[0][0].prior.natural_parameters()
        tensorconf = {'dtype': tensor.dtype, 'device': tensor.device, 
                      'requires_grad': False}

        if weights is None:
            weights = torch.ones(len(modelset), **tensorconf)
            weights /= len(modelset)
        else:
            weights = torch.tensor(weights, **tensorconf)
        weights_param = _default_param(weights, prior_strength)
        return cls(weights_param, modelset)

    def __init__(self, weights, modelset):
        super().__init__(modelset)
        self.weights = weights

    # Log probability of each components.
    def _log_weights(self):
        lhf = self.weights.likelihood_fn
        nparams = self.weights.natural_form()
        data = torch.eye(len(self.modelset), dtype=nparams.dtype, 
                        device=nparams.device, requires_grad=False)
        stats = lhf.sufficient_statistics(data)
        return lhf(nparams, stats)

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        mf_groups = self.modelset.mean_field_factorization()
        mf_groups[0].append(self.weights)
        return mf_groups

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, labels=None, **kwargs):
        # Per-components weighted log-likelihood.
        #log_weights = self.weights.expected_natural_parameters().view(1, -1)
        log_weights = self._log_weights()[None]
        per_component_exp_llh = self.modelset.expected_log_likelihood(stats,
                                                                      **kwargs)

        # Responsibilities and expected llh.
        if labels is None:
            w_per_component_exp_llh = (per_component_exp_llh + log_weights).detach()
            exp_llh = torch.logsumexp(w_per_component_exp_llh, dim=1).view(-1)
            log_resps = w_per_component_exp_llh.detach() - exp_llh.view(-1, 1)
            resps = log_resps.exp()
        else:
            resps = onehot(labels, len(self.modelset),
                            dtype=log_weights.dtype, device=log_weights.device)
            exp_llh = (per_component_exp_llh * resps).sum(dim=-1)

        # Store the responsibilites to accumulate the statistics.
        self.cache['resps'] = resps

        return exp_llh

    def accumulate(self, stats):
        resps = self.cache['resps']
        resps_stats = self.weights.likelihood_fn.sufficient_statistics(resps)
        retval = {
            self.weights: resps_stats.sum(dim=0),
            **self.modelset.accumulate(stats, resps)
        }
        return retval


    ####################################################################
    # DiscreteLatentModel interface.
    ####################################################################

    def posteriors(self, data):
        stats = self.modelset.sufficient_statistics(data)
        log_weights = self.weights.expected_natural_parameters().view(1, -1)
        per_component_exp_llh = self.modelset.expected_log_likelihood(stats)
        per_component_exp_llh += log_weights
        lognorm = torch.logsumexp(per_component_exp_llh, dim=1).view(-1)
        return torch.exp(per_component_exp_llh - lognorm.view(-1, 1))
