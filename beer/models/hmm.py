
'Bayesian HMM model.'

import math
import torch
import numpy as np
from .bayesmodel import DiscreteLatentBayesianModel, BayesianModelSet
from .parameters import ConstantParameter
from ..utils import onehot, logsumexp


class HMM(DiscreteLatentBayesianModel):
    ''' Hidden Markov Model.

    Attributes:
        init_states  (list): Indices of the initial states with
            non-zero probability.
        final_states  (list): Indices of the final states wiht
            non-zero probability.
        trans_mat (``torch.Tensor``): Transition matrix of the HMM states.
        modelset (:any:`BayesianModelSet`): Set of emission densities.

    '''

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

    @staticmethod
    def create_trans_mat(unigram, nstate_per_unit, gamma):
        '''Create a transition matrix

        Args:
            unigram (``torch.Tensor``): Unigram probability of each unit.
            nstate_per_unit (int): Number of states for each unit.
            gamma (float): Insertion penalty, probability of staying in
                the last state of a unit.
        '''

        trans_mat = torch.zeros((len(unigram) * nstate_per_unit,
                                 len(unigram) * nstate_per_unit))
        initial_states = np.arange(0, len(unigram) * nstate_per_unit,
                                   nstate_per_unit)

        for i, j in enumerate(unigram):
            if nstate_per_unit == 1:
                trans_mat[i, :] += (1 - gamma) * unigram
                trans_mat[i, i] += gamma
            else:
                for n in range(nstate_per_unit-1):
                    trans_mat[i*nstate_per_unit+n,
                              i*nstate_per_unit+n : i*nstate_per_unit+n+2] = .5
                trans_mat[i*nstate_per_unit+nstate_per_unit-1,
                          i*nstate_per_unit+nstate_per_unit-1] = gamma
                trans_mat[i*nstate_per_unit+nstate_per_unit-1,
                          initial_states] = (1 - gamma) * unigram
        return trans_mat

    @staticmethod
    def create_ali_trans_mat(tot_states):
        '''Create align transition matrix for a sequence of units

        Args:
            tot_states (int): length of total number of states of the given
                sequence.
        '''

        trans_mat = torch.diag(torch.ones(tot_states) * .5)
        idx1 = torch.arange(0, tot_states-1, dtype=torch.long)
        idx2 = torch.arange(1, tot_states, dtype=torch.long)
        trans_mat[idx1, idx2] = .5
        trans_mat[-1, -1] = 1.
        return trans_mat

    @staticmethod
    def baum_welch_forward(init_states, trans_mat, llhs):
        init_log_prob = -np.log(len(init_states))
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
        final_log_prob = -np.log(len(final_states))
        log_trans_mat = trans_mat.log()
        log_betas = torch.zeros_like(llhs) - float('inf')
        log_betas[-1, final_states] = final_log_prob
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = logsumexp(log_trans_mat + llhs[i+1] + \
                log_betas[i+1], dim=1).view(-1)
        return log_betas

    @staticmethod
    def viterbi(init_states, final_states, trans_mat, llhs):
        init_log_prob = -np.log(len(init_states))
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
        super().__init__(modelset)
        self.init_states = ConstantParameter(torch.tensor(init_states).long(),
                                             fixed_dtype=True)
        self.final_states = ConstantParameter(torch.tensor(final_states).long(),
                                              fixed_dtype=True)
        self.trans_mat = ConstantParameter(trans_mat)

    def _get_log_posteriors(self, pc_llhs, inference_type):
        init_states, final_states, trans_mat = self.init_states.value, \
            self.final_states.value, self.trans_mat.value
        dtype, device = pc_llhs.dtype, pc_llhs.device
        if inference_type == 'viterbi':
            best_path = HMM.viterbi(init_states, final_states, trans_mat, pc_llhs)
            lposteriors = onehot(best_path, len(self.modelset), dtype=dtype,
                                   device=device).log()
        elif inference_type == 'baum_welch':
            log_alphas = HMM.baum_welch_forward(init_states, trans_mat, pc_llhs)
            log_betas = HMM.baum_welch_backward(final_states, trans_mat, pc_llhs)
            lognorm = logsumexp((log_alphas + log_betas)[0].view(-1, 1), dim=0)
            lposteriors = log_alphas + log_betas - lognorm.view(-1, 1)
        else:
            raise ValueError('Unknown inference type: {}'.format(inference_type))
        return lposteriors

    def decode(self, data):
        stats = self.sufficient_statistics(data)
        pc_llhs = self.modelset.expected_log_likelihood(stats)
        best_path = HMM.viterbi(self.init_states.value,
                                self.final_states.value,
                                self.trans_mat.value, pc_llhs)
        return best_path

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, inference_type='baum_welch'):
        pc_exp_llh = self.modelset.expected_log_likelihood(stats)
        log_resps = self._get_log_posteriors(pc_exp_llh.detach(),
                                             inference_type)
        resps = log_resps.exp()
        exp_llh = (pc_exp_llh * resps).sum(dim=-1)
        self.cache['resps'] = resps

        # We ignore the KL divergence term. This may bias the
        # lower-bound a little bit but will not affect the training.
        return exp_llh

    def accumulate(self, stats, parent_msg=None):
        retval = {
            **self.modelset.accumulate(stats, self.cache['resps'])
        }
        return retval

    ####################################################################
    # DiscreteLatentBayesianModel interface.
    ####################################################################

    def posteriors(self, data, inference_type='viterbi'):
        stats = self.modelset.sufficient_statistics(data)
        pc_exp_llh = self.modelset.expected_log_likelihood(stats)
        return self._get_log_posteriors(pc_exp_llh.detach(),
                                        inference_type).exp()


class AlignModelSet(BayesianModelSet):

    def __init__(self, model_set, state_ids):
        '''Args:
        model_set: (:any:`BayesianModelSet`): Set of emission density.
        state_ids (list): sequence of state ids.

        '''
        super().__init__()
        self.modelset = model_set
        self.state_ids = ConstantParameter(torch.tensor(state_ids).long(),
                                           fixed_dtype=True)
        self._idxs = ConstantParameter(list(range(len(state_ids))),
                                       fixed_dtype=True)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        state_ids, idxs = self.state_ids.value, self._idxs.value
        dtype, device = stats.dtype, stats.device
        pc_exp_llh = self.modelset.expected_log_likelihood(stats)
        new_pc_exp_llh = torch.zeros((len(stats), len(state_ids)),
                                     dtype=dtype, device=device)
        new_pc_exp_llh[:, idxs] = pc_exp_llh[:, state_ids[idxs]]
        return new_pc_exp_llh

    def accumulate(self, stats, resps):
        state_ids = self.state_ids.value
        new_resps = torch.zeros((len(stats), len(self.modelset)),
                                 dtype=resps.dtype, device=resps.device)
        for key, val in enumerate(resps.t()):
            new_resps[:, state_ids[key]] += val
        return self.modelset.accumulate(stats, new_resps)

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        '''Args:
        key (int): state index.

        '''
        return self.modelset[self.state_ids.value[key]]

    def __len__(self):
        return len(self.state_ids.value)


__all__ = ['HMM', 'AlignModelSet']
