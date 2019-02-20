'Bayesian Mixture model.'

from operator import mul
import torch
from .basemodel import DiscreteLatentModel
from .parameters import BayesianParameter
from ..dists import Dirichlet, DirichletStdParams
from ..utils import onehot


__all__ = ['Mixture']


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
        prior_nparams = mf_groups[0][0].prior.natural_parameters()
        dtype, device = prior_nparams.dtype, prior_nparams.device

        if weights is None:
            weights = torch.ones(len(modelset), dtype=dtype, device=device)
            weights /= len(modelset)
        else:
            weights = torch.tensor(weights, dtype=dtype, device=device,
                                   requires_grad=False)
        params = DirichletStdParams(prior_strength * weights)
        prior_weights = Dirichlet(params)
        params = DirichletStdParams(prior_strength * weights)
        posterior_weights = Dirichlet(params)
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

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        mf_groups = self.modelset.mean_field_factorization()
        mf_groups[0].append(self.weights)
        return mf_groups

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, labels=None, **kwargs):
        # Per-components weighted log-likelihood.
        log_weights = self.weights.expected_natural_parameters().view(1, -1)
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
        log_weights = self.weights.expected_natural_parameters().view(1, -1)
        per_component_exp_llh = self.modelset.expected_log_likelihood(stats)
        per_component_exp_llh += log_weights

        lognorm = torch.logsumexp(per_component_exp_llh, dim=1).view(-1)
        return torch.exp(per_component_exp_llh - lognorm.view(-1, 1))

    ####################################################################
    # Super-Vector representation interface.
    #################################################################### 

    #def svector_dim(self):
    #    mset_svector_dim = mul(*self.modelset.svector_dim())
    #    return mset_svector_dim + len(self.modelset) - 1
    
    def svector_acc_stats(self):
        _, idxs = self.weights.expected_value().sort(descending=True)
        w_stats = self.weights.posterior.natural_parameters() \
            - self.weights.prior.natural_parameters()
        w_stats = w_stats[idxs]
        w_stats[-1] = w_stats.sum()
        c_stats = self.modelset.svector_acc_stats()[idxs, :]
        return torch.cat([
            c_stats.reshape(-1),
            w_stats
        ])

    def svectors_from_rvectors(self, rvectors):
        ncomps = len(self.modelset)
        comp_rvectors = rvectors[:, :-(ncomps - 1)]
        w_rvectors = rvectors[:, -(ncomps - 1):]
        comp_svectors = self.modelset.svectors_from_rvectors(comp_rvectors)

        # Stable implementation of the log-normalizer of a categorical
        # distribution: ln Z = ln(1 + \sum_i^{D-1} \exp \mu_i)
        # Naive python implementation:
        #   w_lognorm = torch.log(1 + w_rvectors.exp())
        tmp = (1. + torch.logsumexp(w_rvectors, dim=-1))
        w_lognorm = torch.nn.functional.softplus(tmp)
        w_svectors = torch.cat([w_rvectors, w_lognorm.view(-1, 1)], dim=-1)

        return torch.cat([
            comp_svectors.reshape(len(rvectors), -1),
            w_svectors.reshape(len(rvectors), -1)
        ], dim=-1)

    def svector_log_likelihood(self, svectors, acc_stats):
        ncomps = len(self.modelset)
        comp_svectors = svectors[:, :-ncomps]
        comp_svectors = comp_svectors.reshape(len(svectors), ncomps, -1)
        w_svectors = svectors [:, -ncomps:]
        comp_acc_stats = acc_stats[:-ncomps]
        comp_acc_stats = comp_acc_stats.reshape(ncomps, -1)
        w_acc_stats = acc_stats [-ncomps:]
        pc_llhs =  self.modelset.svector_log_likelihood(comp_svectors, 
                                                       comp_acc_stats)
        return pc_llhs.sum(dim=-1) + w_svectors @ w_acc_stats
