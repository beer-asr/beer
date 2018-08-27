
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
        graph (:any:`Graph`): The Graph of the dynamics of the HMM.
        modelset (:any:`BayesianModelSet`): Set of emission densities.

    '''

    @classmethod
    def create(cls, graph, modelset):
        '''Create a :any:`HMM` model.

        Args:
            graph (:any:`Graph`): The Graph of the dynamics of the HMM.
            modelset (:any:`BayesianModelSet`): Set of emission density.

        Returns:
            :any:`HMM`

        '''
        return cls(graph, modelset)

    def __init__(self, graph, modelset):
        '''
        Args:
            graph (:any:`Graph`): The Graph of the dynamics of the HMM.
            modelset (:any:`BayesianModelSet`): Set of emission density.

        '''
        super().__init__(modelset)
        self.graph = ConstantParameter(graph)

    def decode(self, data, inference_graph=None):
        # Prepare the inference graph.
        if inference_graph is None:
            inference_graph = self.graph.value

        # Eventual re-mapping of the pdfs.
        if inference_graph.pdf_id_mapping is not None:
            emissions = AlignModelSet(self.modelset,
                                      inference_graph.pdf_id_mapping)
        else:
            emissions = self.modelset

        stats = self.sufficient_statistics(data)
        pc_llhs = emissions.expected_log_likelihood(stats)
        best_path = inference_graph.best_path(pc_llhs)
        if inference_graph.pdf_id_mapping is not None:
            best_path = [inference_graph.pdf_id_mapping[state]
                         for state in best_path]
            best_path = torch.LongTensor(best_path)
        return best_path


    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, inference_graph=None,
                                inference_type='viterbi', state_path=None):
        if inference_graph is None:
            inference_graph = self.graph.value
        emissions = AlignModelSet(self.modelset, inference_graph.pdf_id_mapping)
        pc_llhs = emissions.expected_log_likelihood(stats)
        if state_path is not None:
            resps = onehot(state_path, inference_graph.n_states,
                           dtype=pc_llhs.dtype, device=pc_llhs.device)
        elif inference_type == 'baum_welch':
            resps = inference_graph.posteriors(pc_llhs)
        elif inference_type == 'viterbi':
            resps = onehot(inference_graph.best_path(pc_llhs),
                           inference_graph.n_states,
                           dtype=pc_llhs.dtype, device=pc_llhs.device)
        else:
            raise ValueError('Unknown inference type {} for the ' \
                             'HMM'.format(inference_type))
        exp_llh = (pc_llhs * resps).sum(dim=-1)

        # Needed to accumulate the statistics.
        self.cache['emissions'] = emissions
        self.cache['resps'] = resps

        # We ignore the KL divergence term. It biases the
        # lower-bound (it may decrease) a little bit but will not affect
        # the value of the parameters.
        return exp_llh

    def accumulate(self, stats, parent_msg=None):
        retval = {
            **self.cache['emissions'].accumulate(stats, self.cache['resps'])
        }
        return retval

    ####################################################################
    # DiscreteLatentBayesianModel interface.
    ####################################################################

    def posteriors(self, data, inference_graph=None):
        if inference_graph is None:
            inference_graph = self.graph.value
        stats = self.modelset.sufficient_statistics(data)
        emissions = AlignModelSet(self.modelset, inference_graph.pdf_id_mapping)
        pc_exp_llh = emissions.expected_log_likelihood(stats)
        return inference_graph.posteriors(pc_exp_llh)


class AlignModelSet(BayesianModelSet):

    def __init__(self, modelset, state_ids):
        '''Args:
        model_set: (:any:`BayesianModelSet`): Set of emission density.
        state_ids (list): sequence of state ids.

        '''
        super().__init__()
        self.modelset = modelset
        self.state_ids = state_ids

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        state_ids = self.state_ids
        dtype, device = stats.dtype, stats.device
        pc_exp_llh = self.modelset.expected_log_likelihood(stats)
        new_pc_exp_llh = torch.zeros((len(stats), len(state_ids)),
                                     dtype=dtype, device=device)
        new_pc_exp_llh[:, :] = pc_exp_llh[:, state_ids]
        return new_pc_exp_llh

    def accumulate(self, stats, resps):
        state_ids = self.state_ids
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
        return self.modelset[self.state_ids[key]]

    def __len__(self):
        return len(self.state_ids)


__all__ = ['HMM', 'AlignModelSet']
