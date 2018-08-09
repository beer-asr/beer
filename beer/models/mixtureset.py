
from collections import namedtuple
import torch
from .bayesmodel import BayesianParameterSet, BayesianParameter
from .bayesmodel import BayesianModelSet
from ..expfamilyprior import DirichletPrior
from ..utils import logsumexp


MixtureSetElement = namedtuple('MixtureSetElement', ['weights', 'modelset'])

class MixtureSet(BayesianModelSet):

    def __init__(self, prior_weights, posterior_weights, modelset):
        super().__init__()
        self.num_mix = len(prior_weights)
        self.num_comp = int(len(modelset) / self.num_mix)
        self.mix_weights = BayesianParameterSet([
            BayesianParameter(prior_weights[i], posterior_weights[i])
            for i in range(self.num_mix)])
        self.modelset = modelset

    @classmethod
    def create(cls, modelset, weights, pseudo_counts=1.):
        prior_weights = [DirichletPrior(pseudo_counts * j) for j in weights]
        posterior_weights = [DirichletPrior(pseudo_counts * j) for j in weights]
        return cls(prior_weights, posterior_weights, modelset)

    def log_weights(self):
        first_value = self.mix_weights[0].expected_value()
        dtype, device = first_value.dtype, first_value.device
        weights = torch.zeros(self.num_mix, self.num_comp, dtype=dtype,
                              device=device)
        for i in range(self.num_mix):
            weight = self.mix_weights[i].expected_value()
            weights[i] = weight
        return weights.reshape(1, self.num_mix, self.num_comp)

    def _local_kl_divergence(self, log_weights, log_resps):
        retval = torch.sum(log_resps.exp() * (log_resps - log_weights), dim=-1)
        return retval

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        mf_groups = self.modelset.mean_field_factorization()
        for weight_param in self.mix_weights:
            mf_groups[0].append(weight_param)
        return mf_groups

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def forward(self, s_stats):
        log_weights = self.log_weights()
        pc_exp_llhs = self.modelset(s_stats)
        pc_exp_llhs = pc_exp_llhs.reshape(-1, self.num_mix, self.num_comp)
        w_pc_exp_llhs = pc_exp_llhs + log_weights

        # Responsibilities.
        log_norm = logsumexp(w_pc_exp_llhs.detach(), dim=-1)
        log_resps = w_pc_exp_llhs.detach() - log_norm[:, :, None]
        resps = log_resps.exp()
        self.cache['resps'] = resps

        # expected llh.
        local_kl_div = self._local_kl_divergence(log_weights, log_resps)
        exp_llh = (pc_exp_llhs * resps).sum(dim=-1)

        return exp_llh - local_kl_div

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        ret_val = {}
        joint_resps = self.cache['resps'] * parent_msg[:,:, None]
        sum_joint_resps = joint_resps.sum(dim=0)
        ret_val = dict(zip(self.mix_weights, sum_joint_resps))
        acc_stats = self.modelset.accumulate(s_stats,
            joint_resps.reshape(-1, self.num_mix * self.num_comp))
        ret_val = {**ret_val, **acc_stats}
        return ret_val

    def __getitem__(self, key):
        weights = torch.exp(self.log_weights()).reshape(self.num_mix,
                                                        self.num_comp)
        mdlset = [self.modelset[i] for i in range(key * self.num_comp,
                  (key+1) * self.num_comp)]
        return MixtureSetElement(weights=weights[key] / weights[key].sum(),
                                 modelset=mdlset)

    def __len__(self):
        return self.num_mix


class SharedModelSet(BayesianModelSet):
    '''Specific model where an internal model set is duplicated
    K times. This model is used with MixtureSet model where all
    the components of the mixtures are shared.

    '''

    def __init__(self, modelset, n_duplicate):
        '''
        Args:
            modelset: (:any:`BayesianModelSet`): Set of densities.
            n_duplicate (int): Number of times to duplicate the model.

        '''
        super().__init__()
        self._modelset = modelset
        self.n_duplicate = n_duplicate

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self._modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self._modelset.sufficient_statistics(data)

    def forward(self, s_stats):
        pc_exp_llh = self._modelset(s_stats)
        return torch.cat([pc_exp_llh] * self.n_duplicate, dim=-1)

    def accumulate(self, s_stats, parent_msg=None):
        s_stats = s_stats
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        new_weights = weights.reshape(-1, self.n_duplicate, len(self._modelset))
        new_weights = new_weights.sum(dim=1)
        return self._modelset.accumulate(s_stats, parent_msg=new_weights)

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        '''Args:
        key (int): state index.

        '''
        new_key = key // self.n_duplicate
        return self._modelset[new_key]

    def __len__(self):
        return len(self._modelset) * self.n_duplicate




def create(model_conf, mean, variance, create_model_handle, modelset=None):
    dtype, device = mean.dtype, mean.device
    n_mix = model_conf['size']
    if modelset is None:
        model_conf['components']['size'] *= n_mix
        modelset = create_model_handle(model_conf['components'], mean, variance)
    n_element = len(modelset) // n_mix
    weights = torch.ones(n_element, dtype=dtype, device=device) / n_element
    weights = weights.repeat(n_mix, 1)
    prior_strength = model_conf['prior_strength']
    prior_weights = [DirichletPrior(prior_strength * w) for w in weights]
    posterior_weights = [DirichletPrior(prior_strength * w) for w in weights]
    return MixtureSet(prior_weights, posterior_weights, modelset)


__all__ = ['MixtureSet', 'SharedModelSet']
