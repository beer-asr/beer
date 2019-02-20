from operator import mul
import torch
from .basemodel import DiscreteLatentModel
from .modelset import DynamicallyOrderedModelSet
from ..utils import onehot


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
        else:
            retval = inference_graph.posteriors(pc_llhs,
                                                trans_posteriors=trans_posteriors)
        return retval

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, inference_graph=None,
                                viterbi=True, state_path=None,
                                scale=1.):
        if inference_graph is None:
            inference_graph = self.graph
        pc_llhs = scale * self._pc_llhs(stats, inference_graph)
        trans_posts = True if inference_graph is None else False
        all_resps = self._inference(pc_llhs, inference_graph, viterbi=viterbi,
                                    state_path=state_path,
                                    trans_posteriors=trans_posts)
        if trans_posts:
            self.cache['resps'], self.cache['trans_resps'] = all_resps
        else:
            self.cache['resps'] = all_resps
        exp_llh = (pc_llhs * self.cache['resps']).sum(dim=-1)
        self.cache['scale'] = scale

        # We ignore the KL divergence term. It biases the
        # lower-bound (it may decrease) a little bit but will not affect
        # the value of the parameters.
        return exp_llh #- kl_div

    def accumulate(self, stats, parent_msg=None):
        scaled_resps = self.cache['scale'] * self.cache['resps']
        retval = {**self.modelset.accumulate(stats, scaled_resps)}
        # By default, we don't do anything with the transition probabilities
        return retval

    ####################################################################
    # DiscreteLatentBayesianModel interface.
    ####################################################################

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

    def posteriors(self, data, inference_graph=None):
        if inference_graph is None:
            inference_graph = self.graph
        stats = self.modelset.sufficient_statistics(data)
        pc_llhs = self._pc_llhs(stats, inference_graph)
        return self._inference(pc_llhs, inference_graph)

    ####################################################################
    # Super-Vector representation interface.
    #################################################################### 

    def svector_dim(self):
        mset_svector_dim = self.modelset.svector_dim()
        return mset_svector_dim + len(self.modelset) - 1
    
    def svector_acc_stats(self):
        _, idxs = self.weights.expected_value().sort(descending=True)
        w_stats = self.weights.posterior.natural_parameters() \
            - self.weights.prior.natural_parameters()
        w_stats = w_stats[idxs]
        w_stats[-1] = w_stats.sum()
        c_stats = self.modelset.svector_acc_stats()[idxs, :]
        return torch.cat([
            c_stats.reshape(-1),
            w_stats
        ])

    def svectors_from_rvectors(self, rvectors):
        ncomps = len(self.modelset)
        comp_rvectors = rvectors[:, :-(ncomps - 1)]
        w_rvectors = rvectors[:, -(ncomps - 1):]
        comp_svectors = self.modelset.svectors_from_rvectors(comp_rvectors)

        # Stable implementation of the log-normalizer of a categorical
        # distribution: ln Z = ln(1 + \sum_i^{D-1} \exp \mu_i)
        # Naive python implementation:
        #   w_lognorm = torch.log(1 + w_rvectors.exp())
        tmp = (1. + torch.logsumexp(w_rvectors, dim=-1))
        w_lognorm = torch.nn.functional.softplus(tmp)
        w_svectors = torch.cat([w_rvectors, w_lognorm.view(-1, 1)], dim=-1)

        return torch.cat([
            comp_svectors.reshape(len(rvectors), -1),
            w_svectors.reshape(len(rvectors), -1)
        ], dim=-1)

    def svector_log_likelihood(self, svectors, acc_stats):
        ncomps = len(self.modelset)
        comp_svectors = svectors[:, :-ncomps]
        comp_svectors = comp_svectors.reshape(len(svectors), ncomps, -1)
        w_svectors = svectors [:, -ncomps:]
        comp_acc_stats = acc_stats[:-ncomps]
        comp_acc_stats = comp_acc_stats.reshape(ncomps, -1)
        w_acc_stats = acc_stats [-ncomps:]
        pc_llhs =  self.modelset.svector_log_likelihood(comp_svectors, 
                                                       comp_acc_stats)
        return pc_llhs.sum(dim=-1) + w_svectors @ w_acc_stats


__all__ = ['HMM']
