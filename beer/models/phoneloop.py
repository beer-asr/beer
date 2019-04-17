
import torch
from .hmm import HMM
from .mixture import _default_param
from .parameters import ConjugateBayesianParameter
from ..dists import Dirichlet, DirichletStdParams
from ..utils import logsumexp


__all__ = ['PhoneLoop']


class PhoneLoop(HMM):
    'Phone Loop HMM.'

    @classmethod
    def create(cls, graph, start_pdf, end_pdf, modelset, weights=None,
               prior_strength=1.0):
        '''Create a PhoneLoop model.

        Args:
            graph (:any:`CompiledGraph`): Decoding graph of the
                phone-loop.
            start_pdf (dict): Mapping symbol/start state of the
                corresponding sub-HMM.
            end_pdf (dict): Mapping symbol/end state of the
                corresponding sub-HMM.
            weights (:any:`BayesianParameter`): Initial unigram
                probability of each phone.
            prior_strength (float): Strength of the prior over the
                weights.
        '''
        # We look at one parameter to check the type of the model.
        bayes_param = modelset.mean_field_factorization()[0][0]
        tensor = bayes_param.prior.natural_parameters()
        dtype, device = tensor.dtype, tensor.device

        if weights is None:
            weights = torch.ones(len(start_pdf), dtype=dtype, device=device)
            weights /= len(start_pdf)
        else:
            weights = torch.tensor(weights, dtype=dtype, device=device,
                                   requires_grad=False)
        weights_param = _default_param(weights, prior_strength)
        return cls(graph, modelset, start_pdf, end_pdf, weights_param)

    def __init__(self, graph, modelset, start_pdf, end_pdf, weights):
        super().__init__(graph, modelset)
        self.start_pdf = start_pdf
        self.end_pdf = end_pdf
        self.weights = weights
        self.weights.register_callback(self._on_weights_update)
        self._on_weights_update()

    def _on_weights_update(self):
        lhf = self.weights.likelihood_fn
        nparams = self.weights.natural_form()
        data = torch.eye(len(self.start_pdf), dtype=nparams.dtype,
                         device=nparams.device, requires_grad=False)
        stats = lhf.sufficient_statistics(data)
        log_weights = lhf(nparams, stats)
        start_idxs = [value for value in self.start_pdf.values()]
        for end_idx in self.end_pdf.values():
            self.graph.trans_log_probs[end_idx, start_idxs] = log_weights

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        params = super().mean_field_factorization()
        params[0] += [self.weights]
        return params

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
            lhf = self.weights.likelihood_fn
            resps_stats = lhf.sufficient_statistics(phone_resps.view(1, -1))
            retval.update({self.weights: resps_stats.view(-1)})
        else:
            nparams = self.weights.posterior.natural_parameters()
            fake_stats = torch.zeros_like(nparams, requires_grad=False)
            retval.update({self.weights: fake_stats})
        return retval

