from operator import mul
import torch
from .basemodel import DiscreteLatentModel
from .modelset import DynamicallyOrderedModelSet
from ..utils import onehot


__all__ = ['HMM']


class HMM(DiscreteLatentModel):
    'Hidden Markov Model with fixed transition probabilities.'

    # The "create" function does nothing in particular but we add it
    # anyway to fit other model constructin pattern.
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
        super().__init__(DynamicallyOrderedModelSet(modelset))
        self.graph = graph

    def _pc_llhs(self, stats, inference_graph):
        order = inference_graph.pdf_id_mapping
        return self.modelset.expected_log_likelihood(stats, order)

    def _inference(self, pc_llhs, inference_graph, viterbi=False,
                   state_path=None, trans_posteriors=False):
        if viterbi or state_path is not None:
            if state_path is None:
                path = inference_graph.best_path(pc_llhs)
            else:
                path = state_path
            posts = onehot(path, inference_graph.n_states,
                           dtype=pc_llhs.dtype, device=pc_llhs.device)
            if trans_posteriors:
                n_states = inference_graph.n_states
                trans_posts = torch.zeros(len(pc_llhs) - 1, n_states, n_states)
                for i, transition in enumerate(zip(path[:-1], path[1:])):
                    src, dest = transition
                    trans_posts[i, src, dest] = 1
                retval = posts, trans_posts
            else:
                retval = posts
            llh = pc_llhs[path].sum()
        else:
            retval, llh = inference_graph.posteriors(pc_llhs,
                                                trans_posteriors=trans_posteriors)
        return retval, llh

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, inference_graph=None,
                                viterbi=False, state_path=None,
                                scale=1.):
        trans_posts = True if inference_graph is None else False
        if inference_graph is None:
            inference_graph = self.graph
        pc_llhs = scale * self._pc_llhs(stats, inference_graph)
        all_resps, llh = self._inference(pc_llhs.detach(), inference_graph, viterbi=viterbi,
                                         state_path=state_path,
                                         trans_posteriors=trans_posts)
        if trans_posts:
            self.cache['resps'], self.cache['trans_resps'] = all_resps
        else:
            self.cache['resps'] = all_resps
        exp_llh = (pc_llhs * self.cache['resps']).sum(dim=-1)
        self.cache['scale'] = scale

        # 'exp_llh' is an approximation but allows to compute the
        # gradient of the likelihood w.r.t. the input
        return exp_llh #llh

    def accumulate(self, stats, parent_msg=None):
        scaled_resps = self.cache['scale'] * self.cache['resps']
        retval = {**self.modelset.accumulate(stats, scaled_resps)}

        # By default, we don't do anything with the transition
        # probabilities.
        return retval

    ####################################################################
    # DiscreteLatentModel interface.

    def decode(self, data, inference_graph=None, scale=1.):
        if inference_graph is None:
            inference_graph = self.graph
        stats = self.sufficient_statistics(data)
        pc_llhs = scale * self._pc_llhs(stats, inference_graph)
        best_path = inference_graph.best_path(pc_llhs)
        best_path = [inference_graph.pdf_id_mapping[state]
                     for state in best_path]
        best_path = torch.LongTensor(best_path)
        return best_path

    def posteriors(self, data, inference_graph=None, scale=1.0):
        if inference_graph is None:
            inference_graph = self.graph
        stats = self.modelset.sufficient_statistics(data) * scale
        pc_llhs = self._pc_llhs(stats, inference_graph)
        return self._inference(pc_llhs, inference_graph)[0]

