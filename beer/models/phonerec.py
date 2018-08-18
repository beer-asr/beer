
'Bayesian Phone Recognizer.'

import math
import torch
import numpy as np
from .hmm import HMM, AlignModelSet


class PhoneRecognizer(HMM):
    'HMM based Phone recognizer.'

    @classmethod
    def create(cls, n_units, n_states_per_unit, n_normal_per_state, modelset):
        '''Create a :any:`PhoneRecognizer` model.

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
        best_path = HMM.viterbi(self.init_states,
                                self.final_states,
                                self.trans_mat, pc_llhs)
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


__all__ = ['HMM', 'AlignModelSet']
