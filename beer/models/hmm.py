
'Bayesian Mixture model.'

import math
import torch
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianParameter


def _expand_labels(labels, ncomp):
    retval = torch.zeros(len(labels), ncomp)
    idxs = torch.range(0, len(labels) - 1).long()
    retval[idxs, labels] = 1
    return retval


def _logsumexp(tensor):
    'Equivatent to: scipy.special.logsumexp(tensor, axis=1)'
    tmax, _ = torch.max(tensor, dim=1, keepdim=True)
    return tmax + (tensor - tmax).exp().sum(dim=1, keepdim=True).log()


class HMM(BayesianModel):
    'Bayesian Mixture Model.'

    def __init__(self, init_states, trans_mat, normalset):
        '''Initialie the HMM model.

        Args:
            init_states : List of indices of states whose weight are non-zero.
            trans_mat (Tensor): Transition matrix of HMM states.
            normalset (``NormalSet``): Set of normal distribution.

        '''
        super().__init__()
        self.init_states = init_states
        self.trans_mat = trans_mat
        self.components = normalset

    def sufficient_statistics(self, data):
        return self.components.sufficient_statistics(data)

    # pylint: disable=C0103
    # Invalid method name.
    def sufficient_statistics_from_mean_var(self, mean, var):
        return self.components.sufficient_statistics_from_mean_var(mean, var)

    @staticmethod
    def forward(init_states, trans_mat, llhs):
        init_log_prob = -math.log(len(init_states))
        log_trans_mat = trans_mat.log()
        log_alphas = torch.zeros_like(llhs) - float('inf')
        log_alphas[0, init_states] = llhs[0, init_states] + init_log_prob

        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += _logsumexp(log_alphas[i-1] + log_trans_mat.t()).view(-1)
        return log_alphas

   # @staticmethod
   # def backward(final_states, log_trans_mat, llhs):
   #     log_betas = np.zeros_like(llhs) - np.def main():
   #     log_betas[-1, final_states] = 0.
   #     for i in reversed


    def accumulate(self, s_stats, parent_msg=None):
        retval = {
            self.weights_params: self._resps.sum(dim=0),
            **self.components.accumulate(s_stats, self._resps)
        }
        self._resps = None
        return retval
