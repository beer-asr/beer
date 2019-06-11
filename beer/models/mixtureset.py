
import torch
from .parameters import ConjugateBayesianParameter
from .modelset import ModelSet
from .mixture import Mixture
from .categoricalset import CategoricalSet
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
    def create(cls, size, modelset, prior_strength=1.):
        '''Create a :any:`MixtureSet' model.

        Args:
            size (int): Number of mixtures.
            modelset (:any:`BayesianModelSet`): Set of models for all
                the mixtures. The order of the model in the set defines
                to which mixture they belong. The total size of the
                model set should be: :any:`size` * `n_comp` where
                `n_comp` is the number of component per mixture.
            strength (float): Strength of the prior over the mixtures'
                weights.
        '''
        # We look at one parameter to check the type of the model.
        bayes_param = modelset.mean_field_factorization()[0][0]
        tensor = bayes_param.prior.natural_parameters()
        dtype, device = tensor.dtype, tensor.device

        ncomp_per_mixture = len(modelset) // size
        weights = torch.ones(size, ncomp_per_mixture) / ncomp_per_mixture
        categoricalset = CategoricalSet.create(weights, prior_strength)
        return cls(categoricalset, modelset)

    def __init__(self, categoricalset, modelset):
        super().__init__()
        self.categoricalset = categoricalset
        self.modelset = modelset

    @property
    def n_comp_per_mixture(self):
        'Number of components per mixture'
        return len(self.modelset) // len(self)

    # Log probability of each components.
    def _log_weights(self, tensorconf):
        data = torch.eye(self.n_comp_per_mixture, **tensorconf)
        stats = self.categoricalset.sufficient_statistics(data)
        return self.categoricalset.expected_log_likelihood(stats).t()

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        l1 = self.modelset.mean_field_factorization()
        l2 = self.categoricalset.mean_field_factorization()
        diff = len(l1) - len(l2)
        if diff > 0:
            l2 += [[] for _ in range(abs(diff))]
        else:
            l1 += [[] for _ in range(abs(diff))]
        return [u + v for u, v in zip(l1, l2)]

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        tensorconf = {'dtype': stats.dtype, 'device': stats.device}
        log_weights = self._log_weights(tensorconf)
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
        m_resps = self.cache['resps']
        jointresps = m_resps * resps[:,:, None]
        totalresps = jointresps.reshape(-1, len(self) * self.n_comp_per_mixture)
        shape = jointresps.shape
        jointresps = jointresps.reshape(-1, shape[-1])
        jointresps = self.categoricalset.sufficient_statistics(jointresps)
        jointresps = jointresps.reshape(shape)
        retval = {
            **self.categoricalset.accumulate_from_jointresps(jointresps),
            **self.modelset.accumulate(stats, totalresps)
        }
        return retval

    ####################################################################
    # ModelSet interface.

    def __len__(self):
        return len(self.categoricalset)

    def __getitem__(self, key):
        ncpm = self.n_comp_per_mixture
        if isinstance(key, int):
            s = slice(key * ncpm, (key + 1) * ncpm)
            return Mixture(self.categoricalset[key], self.modelset[s])
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start * ncpm
            stop = len(self) if key.stop is None else key.stop * ncpm
            step = 1 if key.step is None else key.step * ncpm
            new_s = slice(start, stop, step)
            return self.__class__(self.categoricalset[key],
                                  self.modelset[new_s])
        raise IndexError(f'Unsupported index: {key}')

