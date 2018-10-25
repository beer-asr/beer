
import torch
from .bayesmodel import DiscreteLatentBayesianModel
from .modelset import DynamicallyOrderedModelSet
from .parameters import ConstantParameter
from ..utils import onehot


class HMM(DiscreteLatentBayesianModel):
    ''' Hidden Markov Model.

    Attributes:
        graph (:any:`CompiledGraph`): The (compiled) graph of the
            dynamics of the HMM.
        modelset (:any:`BayesianModelSet`): Set of emission densities.

    '''

    @classmethod
    def create(cls, graph, modelset):
        '''Create a :any:`HMM` model.

        Args:
            graph (:any:`CompiledGraph`): The compiled grpah of the
                dynamics of the HMM.
            modelset (:any:`BayesianModelSet`): Set of emission
                density.

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
        super().__init__(DynamicallyOrderedModelSet(modelset))
        self.graph = ConstantParameter(graph)

    def _pc_llhs(self, stats, inference_graph):
        order = inference_graph.pdf_id_mapping
        return self.modelset.expected_log_likelihood(stats, order)

    def _inference(self, pc_llhs, inference_graph, viterbi=False):
        if viterbi:
            posts = onehot(inference_graph.best_path(pc_llhs),
                           inference_graph.n_states,
                           dtype=pc_llhs.dtype, device=pc_llhs.device)
        else:
            posts = inference_graph.posteriors(pc_llhs)
        return posts

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, inference_graph=None,
                                viterbi=False, state_path=None):
        if inference_graph is None:
            inference_graph = self.graph.value
        pc_llhs = self._pc_llhs(stats, inference_graph)
        if state_path is not None:
            resps = onehot(state_path, inference_graph.n_states,
                           dtype=pc_llhs.dtype, device=pc_llhs.device)
        else:
            resps = self._inference(pc_llhs, inference_graph, viterbi=viterbi)
        exp_llh = (pc_llhs * resps).sum(dim=-1)
        self.cache['resps'] = resps

        # We ignore the KL divergence term. It biases the
        # lower-bound (it may decrease) a little bit but will not affect
        # the value of the parameters.
        return exp_llh #- kl_div

    def accumulate(self, stats, parent_msg=None):
        retval = {
            **self.modelset.accumulate(stats, self.cache['resps'])
        }
        return retval

    ####################################################################
    # DiscreteLatentBayesianModel interface.
    ####################################################################

    def decode(self, data, inference_graph=None):
        if inference_graph is None:
            inference_graph = self.graph.value
        stats = self.sufficient_statistics(data)
        pc_llhs = self._pc_llhs(stats, inference_graph)
        best_path = inference_graph.best_path(pc_llhs)
        best_path = [inference_graph.pdf_id_mapping[state]
                     for state in best_path]
        best_path = torch.LongTensor(best_path)
        return best_path

    def posteriors(self, data, inference_graph=None):
        if inference_graph is None:
            inference_graph = self.graph.value
        stats = self.modelset.sufficient_statistics(data)
        pc_llhs = self._pc_llhs(stats, inference_graph)
        return self._inference(pc_llhs, inference_graph)


__all__ = ['HMM']

