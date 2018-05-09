
'Bayesian Mixture model.'

import math
import torch
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianParameter



def _logsumexp(tensor, dim=1):
    'Equivatent to: scipy.special.logsumexp(tensor, axis=1)'
    tmax, _ = torch.max(tensor, dim=dim, keepdim=True)
    return tmax + (tensor - tmax).exp().sum(dim=dim, keepdim=True).log()


class HMM(BayesianModel):
    'Bayesian Mixture Model.'

    def __init__(self, init_states, final_states, trans_mat, normalset):
        '''Initialie the HMM model.

        Args:
            init_states : List of indices of states whose weight are non-zero.
            trans_mat (Tensor): Transition matrix of HMM states.
            normalset (``NormalSet``): Set of normal distribution.

        '''
        super().__init__()
        self.init_states = init_states
        self.final_states = final_states
        self.trans_mat = trans_mat
        self.components = normalset
        self._resps = None

    def sufficient_statistics(self, data):
        return self.components.sufficient_statistics(data)

    # pylint: disable=C0103
    # Invalid method name.
    def sufficient_statistics_from_mean_var(self, mean, var):
        return self.components.sufficient_statistics_from_mean_var(mean, var)

    @staticmethod
    def baum_welch_forward(init_states, trans_mat, llhs):
        init_log_prob = -math.log(len(init_states))
        log_trans_mat = trans_mat.log()
        log_alphas = torch.zeros_like(llhs) - float('inf')
        log_alphas[0, init_states] = llhs[0, init_states] + init_log_prob

        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += _logsumexp(log_alphas[i-1] + log_trans_mat.t()).view(-1)
        return log_alphas

    @staticmethod
    def baum_welch_backward(final_states, trans_mat, llhs):
        final_log_prob = -math.log(len(final_states))
        log_trans_mat = trans_mat.log()
        log_betas = torch.zeros_like(llhs) - float('inf')
        log_betas[-1, final_states] = final_log_prob
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = _logsumexp(log_trans_mat + llhs[i+1] + \
                log_betas[i+1]).view(-1)
        return log_betas

    @staticmethod
    def viterbi(init_states, final_states, trans_mat, llhs):
        init_log_prob = -math.log(len(init_states))
        backtrack = torch.zeros_like(llhs, dtype=torch.long)
        omega = torch.zeros(llhs.shape[1]).type(llhs.type()) - float('inf')
        omega[init_states] = llhs[0, init_states] + init_log_prob
        log_trans_mat = trans_mat.log()

        for i in range(1, llhs.shape[0]):
            hypothesis = omega + log_trans_mat.t()
            backtrack[i] = torch.argmax(hypothesis, dim=1)
            omega = llhs[i] + hypothesis[range(len(log_trans_mat)), backtrack[i]]
        
        path = [final_states[torch.argmax(omega[final_states])]]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])
        return torch.LongTensor(path)


    def forward(self, s_stats, labels=None):
        pc_exp_llh = self.components(s_stats)
        log_alphas = HMM.baum_welch_forward(self.init_states, self.trans_mat, pc_exp_llh)
        log_betas = HMM.baum_welch_backward(self.final_states, self.trans_mat, pc_exp_llh)

        if labels is not None:
            raise NotImplementedError
        else:
            exp_llh = _logsumexp((log_alphas + log_betas)[0], dim=0)
            self._resps = torch.exp(log_alphas + log_betas - exp_llh.view(-1, 1))
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        retval = {
            **self.components.accumulate(s_stats, self._resps)
        }
        self._resps = None
        return retval
