
'Bayesian Phone Recognizer.'

import math
import torch
import numpy as np
from .hmm import HMM, AlignModelSet
from ..utils import onehot


class PhoneRecognizer(HMM):
    'HMM based Phone recognizer.'

    def __init__(self, acoustic_graph, emissions):
        '''
        Args:
            acoustic_graph  (:any:`AcousticGraph`): Decoding acoustic
                graph.
            emissions  (:any:`BayesianModelSet`): Set of emissions for
                each state of the graph.
        '''
        super().__init__(emissions)
        self.acoustic_graph = acoustic_graph
        self.emissions = emissions

    def decode(self, data):
        stats = self.sufficient_statistics(data)
        pc_llhs = self.emissions.expected_log_likelihood(stats)
        return self.acoustic_graph.best_path(pc_llhs)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.emissions.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.emissions.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, labels=None,
                                inference_type='baum_welch'):
        # If a sequence of labels is provided, change the inference
        # graph.
        if labels not None:
            inference_graph = self.acoustic_graph.alignment_graph(labels)
            #emissions = self.AlignModelSet()
        else:
            inference_graph = self.acoustic_graph
            emissions = self.emissions

        pc_llhs = emissions.expected_log_likelihood(stats)

        if inference_type == 'baum_welch':
            resps = inference_graph(pc_llhs)
        elif inference_type == 'viterbi':
            best_path = inference_graph.best_path()
            resps = onehot(best_path, len(emissions),
                           dtype=pc_llhs.dtype, device=pc_llhs.device)
        else:
            raise ValueError('Unknown inference type: {}'.format(inference_type))

        # Store the responsibilities to compute the natural gradients.
        self.cache['resps'] = resps
        self.cache['emissions'] = emissions

        exp_llh = (pc_llhs * resps).sum(dim=-1)

        # We ignore the KL divergence term. This may bias the
        # lower-bound a little bit but will not affect the training.
        return exp_llh

    def accumulate(self, stats):
        return self.cache['emissions'].accumulate(stats, self.cache['resps'])

    ####################################################################
    # DiscreteLatentBayesianModel interface.
    ####################################################################

    def posteriors(self, data):
        stats = self.sufficient_statistics(data)
        pc_llhs = self.emissions.expected_log_likelihood(stats)
        return self.acoustic_graph.posteriors(pc_llhs)


__all__ = ['PhoneRecognizer']
