
from collections import namedtuple
import torch
from .bayesmodel import BayesianParameterSet, BayesianParameter
from .bayesmodel import BayesianModelSet
from ..expfamilyprior import DirichletPrior
from ..utils import logsumexp


MixtureSetElement = namedtuple('MixtureSetElement', ['weight', 'modelset'])

class MixtureSet(BayesianModelSet):
    def __init__(self, prior_weights, posterior_weights, modelset):
        super().__init__()
        self.num_mix = len(prior_weights)
        self.num_comp = int(len(modelset) / self.num_mix)
        self.mix_weights = BayesianParameterSet([
            BayesianParameter(prior_weights[i], posterior_weights[i])
            for i in range(self.num_mix)])
        self.modelset = modelset
        self._resps = None
    
    @classmethod
    def create(cls, modelset, weights, pseudo_counts=1.):
        prior_weights = [DirichletPrior(pseudo_counts * j) for j in weights]
        posterior_weights = [DirichletPrior(pseudo_counts * j) for j in weights]
        return cls(prior_weights, posterior_weights, modelset)
    
    def log_weights(self): # p(c|k)
        first_value = self.mix_weights[0].expected_value()
        dtype, device = first_value.dtype, first_value.device
        weights = torch.zeros(self.num_mix, self.num_comp, dtype=dtype,
                              device=device)
        for i in range(self.num_mix):
            weight = self.mix_weights[i].expected_value()
            weights[i] = weight
        return weights.reshape(1, self.num_mix, self.num_comp)
        
    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)
    
    def forward(self, s_stats):
        exp_llhs = self.modelset(s_stats).reshape(-1, self.num_mix, self.num_comp)
        exp_llhs += self.log_weights()
        log_norm = logsumexp(exp_llhs, dim=-1)
        self._resps = torch.exp(exp_llhs - log_norm[:, :, None]) 
        return log_norm
        
    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None: # p(k|x)
            raise ValueError('"parent_msg" should not be None')
        ret_val = {} 
        joint_resps = self._resps * parent_msg[:,:, None]
        sum_joint_resps = joint_resps.sum(dim=0)
        ret_val = dict(zip(self.mix_weights, sum_joint_resps))
        acc_stats = self.modelset.accumulate(s_stats, joint_resps.reshape(-1, self.num_mix * self.num_comp))
        ret_val = {**ret_val, **acc_stats}
        return ret_val
    
    def __getitem__(self, key):
        weights = torch.exp(self.log_weights()).sum(dim=2).reshape(self.num_mix)
        mdlset = [self.modelset[i] for i in range(key * self.num_comp,
                  (key+1) * self.num_comp)]
        return MixtureSetElement(weight=weights[key]/weights.sum(), 
                                 modelset=mdlset) 
    
    def expected_natural_params_from_resps(self, resps):
        pass
    
    def __len__(self):
        return self.num_mix
   
    def double(self):
        return self.__class__(
            [weight_param.prior.double() for weight_param in self.mix_weights],
            [weight_param.posterior.double() for weight_param in self.mix_weights],
            self.modelset.double()
        )
    
    def float(self):
        return self.__class__(
            [weight_param.prior.float() for weight_param in self.mix_weights],
            [weight_param.posterior.float() for weight_param in self.mix_weights],
            self.modelset.float()
        )
   
    def to(self, device):
        return self.__class__(
            [weight_param.prior.to(device) for weight_param in self.mix_weights],
            [weight_param.posterior.to(device) for weight_param in self.mix_weights],
            self.modelset.to(device)
        )

def create(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    n_mix = model_conf['size']
    model_conf['components']['size'] *= n_mix
    modelset = create_model_handle(model_conf['components'], mean, variance)
    n_element = int(len(modelset) / n_mix) 
    weights = torch.ones(n_element, dtype=dtype, device=device) / n_element
    weights = weights.repeat(n_mix, 1)
    prior_strength = model_conf['prior_strength']
    prior_weights = [DirichletPrior(prior_strength * j) for j in weights]
    posterior_weights = [DirichletPrior(prior_strength * j) for j in weights]
    return MixtureSet(prior_weights, posterior_weights, modelset)


__all__ = ['MixtureSet']
