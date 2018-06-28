
'Bayesian HMM model.'

import torch
import numpy as np
from .bayesmodel import BayesianModel, BayesianModelSet
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

    def __init__(self, init_states, final_states, trans_mat, modelset, training_type):
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
        self.training_type = training_type

    @classmethod
    def create(cls, init_states, final_states, trans_mat, modelset, training_type):
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
        return cls(init_states, final_states, trans_mat, modelset, training_type)

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def sufficient_statistics_from_mean_var(self, mean, var):
        return self.modelset.sufficient_statistics_from_mean_var(mean, var)

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

    def decode(self, data):
        stats = self.sufficient_statistics(data)
        pc_llhs = self.modelset(stats)
        best_path = HMM.viterbi(self.init_states,
                                self.final_states,
                                self.trans_mat, pc_llhs)
        #phones = convert_state_to_phone(phone_dict, list(best_path.numpy()), nstate_per_phone)
        return best_path

    def float(self):
        return self.__class__(
            self.init_states,
            self.final_states,
            self.trans_mat.float(),
            self.modelset.float(),
            self.training_type
        )

    def double(self):
        return self.__class__(
            self.init_states,
            self.final_states,
            self.trans_mat.double(),
            self.modelset.double(),
            self.training_type
        )

    def to(self, device):
        return self.__class__(
            self.init_states,
            self.final_states,
            self.trans_mat.to(device),
            self.modelset.to(device),
            self.training_type
        )

    def forward(self, s_stats, state_path=None):
        pc_exp_llh = self.modelset(s_stats)
        if state_path is not None:
            onehot_labels = onehot(state_path, len(self.modelset),
                                   dtype=pc_exp_llh.dtype,
                                   device=pc_exp_llh.device)
            exp_llh = (pc_exp_llh * onehot_labels).sum(dim=-1)
            self._resps = onehot_labels
        elif self.training_type == 'viterbi':
            onehot_labels = onehot(HMM.viterbi(self.init_states, 
                                  self.final_states, self.trans_mat, pc_exp_llh), 
                                  len(self.modelset), dtype=pc_exp_llh.dtype,
                                  device=pc_exp_llh.device)
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


class AlignModelSet(BayesianModelSet):

    def __init__(self, model_set, state_ids):
        '''Args:
        model_set: (:any:`BayesianModelSet`): Set of emission density.
        state_ids (list): sequence of state ids.

        '''
        super().__init__()
        self.model_set = model_set
        self.state_ids = torch.tensor(state_ids).long()
        self._idxs = list(range(len(self.state_ids)))

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def sufficient_statistics(self, data):
        return len(data), self.model_set.sufficient_statistics(data)

    def float(self):
        return self.__class__(
            self.model_set.float(),
            self.state_ids
        )

    def double(self):
        return self.__class__(
            self.model_set.double(),
            self.state_ids
        )

    def to(self, device):
        return self.__class__(
            self.model_set.to(device),
            self.state_ids
        )

    def forward(self, len_s_stats):
        length, s_stats = len_s_stats
        pc_exp_llh = self.model_set(s_stats)
        new_pc_exp_llh = torch.zeros((length, len(self.state_ids)),
                                     dtype=pc_exp_llh.dtype, device=pc_exp_llh.device)
        new_pc_exp_llh[:, self._idxs] = pc_exp_llh[:, self.state_ids[self._idxs]]
        return new_pc_exp_llh

    def accumulate(self, len_s_stats, parent_msg=None):
        length, s_stats = len_s_stats
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        new_weights = torch.zeros((length, len(self.model_set)),
                                  dtype=weights.dtype, device=weights.device)
        for key, val in enumerate(weights.t()):
            new_weights[:, self.state_ids[key]] += val

        return self.model_set.accumulate(s_stats, parent_msg=new_weights)

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        '''Args:
        key (int): state index.

        '''
        return self.model_set[self.state_ids[key]]

    def __len__(self):
        return len(self.state_ids)

    # TODO: This is code is to change as it would be more
    # consistent if the object change the responsibilities and
    # give the modified resps to the internal model set.
    def expected_natural_params_as_matrix(self):
        parameters = self.model_set.expected_natural_params_as_matrix()
        return parameters[self.state_ids]

    def expected_natural_params_from_resps(self, resps):
        matrix = self.expected_natural_params_as_matrix()
        return resps @ matrix


__all__ = ['HMM', 'AlignModelSet']
