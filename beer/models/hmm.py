
'Bayesian HMM model.'

import math
import torch
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianParameter
from ..utils import onehot, logsumexp




class HMM(BayesianModel):
    ''' Hidden Markov Model.

    Attributes: 
        init_states  (list): Indices of initial states who have 
            non-zero probability.
        final_states  (list): Indices of final states who have 
            non-zero probability.
        trans_mat (``torch.Tensor``): Transition matrix of HMM states.
        modelset (:any:`BayesianModelSet`): Set of emission density.

    Example:
        >>> # Create a set of Normal densities.
        >>> mean = torch.zeros(2)
        >>> cov = torch.eye(2)
        >>> normalset = beer.NormalSetSharedFullCovariance.create(mean, cov, 3, noise_std=0.1)
        >>> init_state = [0]
        >>> final_state = [1]
        >>> hmm = beer.HMM(init_state, final_state, trans_mat, normalset)
        >>> hmm.init_states
        [0]
        >>> hmm.final_states
        [1]
        >>> hmm.trans_mat
        tensor([[ 0.5000,  0.5000],
                [ 1.0000,  0.0000]])
    '''

    def __init__(self, init_states, final_states, trans_mat, modelset):
        '''
        Args:
            init_states  (list): Indices of initial states who have 
                non-zero probability.
            final_states  (list): Indices of final states who have 
                non-zero probability.
            trans_mat (``torch.Tensor``): Transition matrix of HMM states.
            modelset (:any:`BayesianModelSet`): Set of emission density.

        '''
        super().__init__()
        self.init_states = init_states
        self.final_states = final_states
        self.trans_mat = trans_mat
        self.modelset = modelset
        self._resps = None
    
    @classmethod
    def create(cls, init_states, final_states, trans_mat, modelset):
        '''Create a :any:`HMM` model.

        Args:
            init_states  (list): Indices of initial states who have 
                non-zero probability.
            final_states  (list): Indices of final states who have 
                non-zero probability.
            trans_mat (``torch.Tensor``): Transition matrix of HMM states.
            modelset (:any:`BayesianModelSet`): Set of emission density.

        Returns:
            :any:`HMM`

        '''
        return cls(init_states, final_states, trans_mat, modelset)

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    # pylint: disable=C0103
    # Invalid method name.
    def sufficient_statistics_from_mean_var(self, mean, var):
        return self.modelset.sufficient_statistics_from_mean_var(mean, var)

    @staticmethod
    def baum_welch_forward(init_states, trans_mat, llhs):
        init_log_prob = -math.log(len(init_states))
        log_trans_mat = trans_mat.log()
        log_alphas = torch.zeros_like(llhs) - float('inf')
        log_alphas[0, init_states] = llhs[0, init_states] + init_log_prob

        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += logsumexp(log_alphas[i-1] + log_trans_mat.t(), 
                                       dim=1).view(-1)
        return log_alphas

    @staticmethod
    def baum_welch_backward(final_states, trans_mat, llhs):
        final_log_prob = -math.log(len(final_states))
        log_trans_mat = trans_mat.log()
        log_betas = torch.zeros_like(llhs) - float('inf')
        log_betas[-1, final_states] = final_log_prob
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = logsumexp(log_trans_mat + llhs[i+1] + \
                log_betas[i+1], dim=1).view(-1)
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


    def forward(self, s_stats, latent_variables=None):
        pc_exp_llh = self.modelset(s_stats)
        log_alphas = HMM.baum_welch_forward(self.init_states, self.trans_mat, pc_exp_llh)
        log_betas = HMM.baum_welch_backward(self.final_states, self.trans_mat, pc_exp_llh)

        if latent_variables is not None:
            onehot_labels = onehot(latent_variables, len(self.modelset))
            onehot_labels = onehot_labels.type(pc_exp_llh.type())
            exp_llh = (pc_exp_llh * onehot_labels).sum(dim=-1)
            self._resps = onehot_labels
        else:
            log_alphas = HMM.baum_welch_forward(self.init_states, self.trans_mat, pc_exp_llh)
            log_betas = HMM.baum_welch_backward(self.final_states, self.trans_mat, pc_exp_llh)
            exp_llh = logsumexp((log_alphas + log_betas)[0].view(-1, 1), dim=0)
            self._resps = torch.exp(log_alphas + log_betas - exp_llh.view(-1, 1))
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        retval = {
            **self.modelset.accumulate(s_stats, self._resps)
        }
        self._resps = None
        return retval
