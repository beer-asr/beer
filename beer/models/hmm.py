
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


    def _align(self, pc_llhs, inference_graph, align_type):
        # Align the data.
        if align_type == 'viterbi':
            ali = inference_graph.best_path(pc_llhs)
            ali = onehot(ali, inference_graph.n_states, pc_llhs.dtype,
                         pc_llhs.device)
        elif align_type == 'baum_welch':
            ali = inference_graph.posteriors(pc_llhs)
        elif align_type == 'sample':
            raise NotImplementedError
        else:
            raise ValueError('Unknown alignment type: {}'.format(align_type))

        return ali

    def align(self, data, inference_graph=None, align_type='viterbi'):
        '''Align input data to the HMM states.

        Args:
            data (``torch.Tensor``): Data to align.
            inference_graph (:any:`CompiledGraph`): Alignment graph,
                if none given, use the given graph.
            align_type (string): Type of alignment ('viterbi', 'baum_welch',
                or 'sample').

        Returns:
            alignments (``torch.Matrix``): Alignment matrix.

        '''
        if inference_graph is None:
            inference_graph = self.graph.value
        stats = self.sufficient_statistics(data)
        emissions = AlignModelSet(self.modelset, inference_graph.pdf_id_mapping)
        pc_llhs = emissions.expected_log_likelihood(stats)
        return self._align(pc_llhs, inference_graph, align_type)

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
                                align_type='viterbi', resps=None):
        if inference_graph is None:
            inference_graph = self.graph.value
        emissions = AlignModelSet(self.modelset, inference_graph.pdf_id_mapping)
        pc_llhs = emissions.expected_log_likelihood(stats)
        if resps is None:
            resps = self._align(pc_llhs, inference_graph, align_type)
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
        return self.align(data, inference_graph, align_type='baum_welch')


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
