
from collections import namedtuple
import torch
from .parameters import BayesianParameterSet, BayesianParameter
from .modelset import ModelSet
from .mixture import Mixture
from ..dists import Dirichlet, DirichletStdParams
from ..utils import logsumexp

__all__ = ['MixtureSet']



class MixtureSet(ModelSet):
    '''Set of mixture models, each of them having the same number of
    components.

    '''

    @classmethod
    def create(cls, size, modelset, weights=None, prior_strength=1.):
        '''Create a :any:`MixtureSet' model.

        Args:
            size (int): Number of mixtures.
            modelset (:any:`BayesianModelSet`): Set of models for all
                the mixtures. The order of the model in the set defines
                to which mixture they belong. The total size of the
                model set should be: :any:`size` * `n_comp` where
                `n_comp` is the number of component per mixture.
            prior_strength (float): Strength the prior over the
                weights.

        '''
        # We look at one parameter to check the type of the model.
        bayes_param = modelset.mean_field_factorization()[0][0]
        tensor = bayes_param.prior.natural_parameters()
        dtype, device = tensor.dtype, tensor.device

        n_comp_per_mixture = len(modelset) // size
        if weights is None:
            weights = torch.ones(n_comp_per_mixture, dtype=dtype, device=device)
            weights *= 1. / n_comp_per_mixture
        prior_weights = []
        params = DirichletStdParams(prior_strength * weights)
        prior_weights = Dirichlet(params)
        posterior_weights = []
        for _ in range(size):
            params = DirichletStdParams(prior_strength * weights)
            posterior_weights.append(Dirichlet(params))
        return cls(prior_weights, posterior_weights, modelset)

    def __init__(self, prior_weights, posterior_weights, modelset):
        '''
        Args:
            prior_weights (:any:`Dirichlet`): Prior distribution
                over the weights for each mixture.
            posterior_weights (list of :any:`Dirichlet`): Posterior
                distribution over the weights for each mixture.
            modelset (:any:`BayesianModelSet`): Set of models for all
                mixtures.

        '''
        super().__init__()
        self.weights = BayesianParameterSet([
            BayesianParameter(prior_weights, posterior)
            for posterior in posterior_weights])
        self.modelset = modelset

    def __getitem__(self, key):
        if isinstance(key, slice):
            weights = self.weights[key]
            prior = weights[0].prior
            posteriors = [bayes_param.posterior for bayes_param in weights]
            shift = self.n_comp_per_mixture
            if key.step is not None:
                raise NotImplementedError('specific slice step is not implemented')
            new_slice = slice(
                key.start * shift if key.start is not None else None,
                key.stop * shift if key.stop is not None else None,
                None
            )
            mdlset = self.modelset[new_slice]
            return MixtureSet(prior, posteriors, mdlset)
        weights = self.weights[key]
        start = key * self.n_comp_per_mixture
        stop = (key + 1) * self.n_comp_per_mixture
        mdlset = self.modelset[start:stop]
        return Mixture(weights.prior, weights.posterior, mdlset)

    def __len__(self):
        return len(self.weights)

    @property
    def n_comp_per_mixture(self):
        'Number of components per mixture'
        return len(self.modelset) // len(self)

    ####################################################################
    # Model interface.
    ####################################################################

    def mean_field_factorization(self):
        return [self.modelset.mean_field_factorization()[0] + [*self.weights]]

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        log_weights = self.weights.expected_natural_parameters()
        pc_exp_llhs = self.modelset.expected_log_likelihood(stats)
        pc_exp_llhs = pc_exp_llhs.reshape(-1, len(self), self.n_comp_per_mixture)
        w_pc_exp_llhs = pc_exp_llhs + log_weights[None]

        # Responsibilities.
        log_norm = logsumexp(w_pc_exp_llhs.detach(), dim=-1)
        log_resps = w_pc_exp_llhs.detach() - log_norm[:, :, None]
        resps = log_resps.exp()
        self.cache['resps'] = resps

        # expected llh.
        exp_llh = (pc_exp_llhs * resps).sum(dim=-1)

        # Local KL divergence.
        local_kl_div = torch.sum(resps * (log_resps - log_weights), dim=-1)

        return exp_llh - local_kl_div

    def accumulate(self, stats, resps):
        ret_val = {}
        joint_resps = self.cache['resps'] * resps[:,:, None]
        sum_joint_resps = joint_resps.sum(dim=0)
        ret_val = dict(zip(self.weights, sum_joint_resps))
        acc_stats = self.modelset.accumulate(stats,
            joint_resps.reshape(-1, len(self) * self.n_comp_per_mixture))
        ret_val = {**ret_val, **acc_stats}
        return ret_val

