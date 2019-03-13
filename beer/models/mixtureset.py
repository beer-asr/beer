
import torch
from .parameters import ConjugateBayesianParameter
from .modelset import ModelSet
from .mixture import Mixture
from ..dists import Dirichlet
from ..utils import logsumexp

__all__ = ['MixtureSet']


########################################################################
# Helper to build the default parameters.

def _default_param(weights, prior_strength):
    prior = Dirichlet.from_std_parameters(prior_strength * weights)
    posterior = Dirichlet.from_std_parameters(prior_strength * weights)
    return ConjugateBayesianParameter(prior, posterior)

########################################################################


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
            weights = torch.ones(size, n_comp_per_mixture, dtype=dtype,
                                 device=device)
            weights *= 1. / n_comp_per_mixture
        weights_param = _default_param(weights, prior_strength)
        return cls(weights_param, modelset)

    def __init__(self, weights, modelset):
        super().__init__()
        self.weights = weights
        self.modelset = modelset

    @property
    def n_comp_per_mixture(self):
        'Number of components per mixture'
        return len(self.modelset) // len(self)

    # Log probability of each components.
    def _log_weights(self):
        lhf = self.weights.likelihood_fn
        nparams = self.weights.natural_form()
        data = torch.eye(self.n_comp_per_mixture, dtype=nparams.dtype,
                        device=nparams.device, requires_grad=False)
        stats = lhf.sufficient_statistics(data)
        return lhf(nparams, stats).t()

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        retval = self.modelset.mean_field_factorization()
        retval[0] += [self.weights]
        return retval

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        log_weights = self._log_weights()
        pc_exp_llhs = self.modelset.expected_log_likelihood(stats)
        pc_exp_llhs = pc_exp_llhs.reshape(-1, len(self), self.n_comp_per_mixture)
        w_pc_exp_llhs = pc_exp_llhs + log_weights[None]

        # Responsibilities.
        log_norm = logsumexp(w_pc_exp_llhs.detach(), dim=-1)
        log_resps = w_pc_exp_llhs.detach() - log_norm[:, :, None]
        resps = log_resps.exp()
        self.cache['resps'] = resps

        return log_norm

    def accumulate(self, stats, resps):
        jointresps = self.cache['resps'] * resps[:,:, None]
        lhf = self.weights.likelihood_fn
        jointresps_stats = lhf.sufficient_statistics(jointresps.sum(dim=0))
        totalresps = jointresps.reshape(-1, len(self) * self.n_comp_per_mixture)
        retval = {
            self.weights: jointresps_stats,
            **self.modelset.accumulate(stats, totalresps)
        }
        return retval

    ####################################################################
    # ModelSet interface.

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, key):
        ncpm = self.n_comp_per_mixture
        if isinstance(key, int):
            s = slice(key * ncpm, (key + 1) * ncpm)
            return Mixture(self.weights[key], self.modelset[s])
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start * ncpm
            stop = len(self) if key.stop is None else key.stop * ncpm
            step = 1 if key.step is None else key.step * ncpm
            new_s = slice(start, stop, step)
            return self.__class__(self.weights[key], self.modelset[new_s])
        raise IndexError(f'Unsupported index: {key}')


