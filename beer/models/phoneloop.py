
import torch
from .hmm import HMM
from .categorical import Categorical
from .categoricalset import CategoricalSet
from .parameters import ConjugateBayesianParameter
from ..utils import logsumexp


__all__ = ['PhoneLoop', 'BigramPhoneLoop']


class PhoneLoop(HMM):
    'Phone Loop HMM.'

    @classmethod
    def create(cls, graph, start_pdf, end_pdf, modelset, categorical=None,
               prior_strength=1.0):
        '''Create a PhoneLoop model.

        Args:
            graph (:any:`CompiledGraph`): Decoding graph of the
                phone-loop.
            start_pdf (dict): Mapping symbol/start state of the
                corresponding sub-HMM.
            end_pdf (dict): Mapping symbol/end state of the
                corresponding sub-HMM.
            categorical (``Categorical``): Categorical model of the
                mixing weights.
            prior_strength (float): Strength of the prior over the
                weights.
        '''
        # We look at one parameter to check the type of the model.
        bayes_param = modelset.mean_field_factorization()[0][0]
        tensor = bayes_param.prior.natural_parameters()
        dtype, device = tensor.dtype, tensor.device

        if categorical is None:
            weights = torch.ones(len(start_pdf), dtype=dtype, device=device)
            weights /= len(start_pdf)
            categorical = Categorical.create(weights, prior_strength)
        return cls(graph, modelset, start_pdf, end_pdf, categorical)

    def __init__(self, graph, modelset, start_pdf, end_pdf, categorical):
        super().__init__(graph, modelset)
        self.start_pdf = start_pdf
        self.end_pdf = end_pdf
        self.categorical = categorical
        param = self.categorical.mean_field_factorization()[0][0]
        param.register_callback(self._on_weights_update)
        self._on_weights_update()

    def _on_weights_update(self):
        mean = self.categorical.mean
        tensorconf = {'dtype': mean.dtype, 'device': mean.device,
                      'requires_grad': False}
        data = torch.eye(len(self.start_pdf), **tensorconf)
        stats = self.categorical.sufficient_statistics(data)
        log_weights = self.categorical.expected_log_likelihood(stats)
        start_idxs = [value for value in self.start_pdf.values()]
        for end_idx in self.end_pdf.values():
            self.graph.trans_log_probs[end_idx, start_idxs] = log_weights

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        l1 = self.modelset.mean_field_factorization()
        l2 = self.categorical.mean_field_factorization()
        diff = len(l1) - len(l2)
        if diff > 0:
            l2 += [[] for _ in range(abs(diff))]
        else:
            l1 += [[] for _ in range(abs(diff))]
        return [u + v for u, v in zip(l1, l2)]

    def expected_log_likelihood(self, *args, **kwargs):
        #self._on_weights_update()
        return super().expected_log_likelihood(*args, **kwargs)

    def accumulate(self, stats, parent_msg=None):
        retval = super().accumulate(stats, parent_msg)

        # If the phone loop is trained with forced alignments, we don't
        # train the transitions.
        if 'trans_resps' in self.cache:
            trans_resps = self.cache['trans_resps'].sum(dim=0)
            start_idxs = [value for value in self.start_pdf.values()]
            end_idxs = [value for value in self.end_pdf.values()]
            phone_resps = trans_resps[:, start_idxs]
            phone_resps = phone_resps[end_idxs, :].sum(dim=0)
            phone_resps += self.cache['resps'][0][start_idxs]
            resps_stats = self.categorical.sufficient_statistics(
                            phone_resps.view(1, -1))
            retval.update(self.categorical.accumulate(resps_stats))
        else:
            fake_stats = torch.zeros_like(self.categorical.mean, requires_grad=False)
            retval.update(self.categorical.accumulate(fake_stats[None, :]))
        return retval


class BigramPhoneLoop(HMM):
    'Phone Loop HMM with Bigram phonotactic language model..'

    @classmethod
    def create(cls, graph, start_pdf, end_pdf, modelset, categoricalset=None,
               prior_strength=1.0):
        '''Create a BigramPhoneLoop model.

        Args:
            graph (:any:`CompiledGraph`): Decoding graph of the
                phone-loop.
            start_pdf (dict): Mapping symbol/start state of the
                corresponding sub-HMM.
            end_pdf (dict): Mapping symbol/end state of the
                corresponding sub-HMM.
            categoricalset (``CategoricalSet``): Set of categorical models
                of the mixing weights.
            prior_strength (float): Strength of the prior over the
                weights.
        '''
        # We look at one parameter to check the type of the model.
        bayes_param = modelset.mean_field_factorization()[0][0]
        tensor = bayes_param.prior.natural_parameters()
        dtype, device = tensor.dtype, tensor.device

        if categoricalset is None:
            weights = torch.ones(len(start_pdf), len(start_pdf), dtype=dtype, 
                                 device=device)
            weights /= len(start_pdf)
            categoricalset = CategoricalSet.create(weights, prior_strength)
        return cls(graph, modelset, start_pdf, end_pdf, categoricalset)

    def __init__(self, graph, modelset, start_pdf, end_pdf, categoricalset):
        super().__init__(graph, modelset)
        self.start_pdf = start_pdf
        self.end_pdf = end_pdf
        self.categoricalset = categoricalset

    def _on_weights_update(self):
        mean = self.categoricalset.mean
        tensorconf = {'dtype': mean.dtype, 'device': mean.device,
                      'requires_grad': False}
        data = torch.eye(len(self.start_pdf), **tensorconf)
        stats = self.categoricalset.sufficient_statistics(data)
        log_weights = self.categoricalset.expected_log_likelihood(stats)
        start_idxs = [value for value in self.start_pdf.values()]
        for i, end_idx in enumerate(self.end_pdf.values()):
            loop_prob = self.graph.trans_log_probs[end_idx, end_idx].exp()
            residual_log_prob = (1 - loop_prob).log()
            self.graph.trans_log_probs[end_idx, start_idxs] = \
                    residual_log_prob + log_weights[i]

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        l1 = self.modelset.mean_field_factorization()
        l2 = self.categoricalset.mean_field_factorization()
        diff = len(l1) - len(l2)
        if diff > 0:
            l2 += [[] for _ in range(abs(diff))]
        else:
            l1 += [[] for _ in range(abs(diff))]
        return [u + v for u, v in zip(l1, l2)]

    def expected_log_likelihood(self, *args, **kwargs):
        self._on_weights_update()
        return super().expected_log_likelihood(*args, **kwargs)

    def accumulate(self, stats, parent_msg=None):
        retval = super().accumulate(stats, parent_msg)
        # If the phone loop is trained with forced alignments, we don't
        # train the transitions.
        if 'trans_resps' in self.cache:
            trans_resps = self.cache['trans_resps']#.sum(dim=0)
            start_idxs = [value for value in self.start_pdf.values()]
            end_idxs = [value for value in self.end_pdf.values()]
            phone_resps = trans_resps[:, :, start_idxs]
            phone_resps = phone_resps[:, end_idxs, :]
            resps_stats = self.categoricalset.sufficient_statistics(phone_resps)
            retval.update(self.categoricalset.accumulate_from_jointresps(resps_stats))
        else:
            fake_stats = torch.zeros_like(self.categoricalset.mean, 
                                          requires_grad=False)
            retval.update(self.categoricalset.accumulate(fake_stats[None, :]))
        return retval
