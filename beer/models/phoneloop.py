
import torch
from .hmm import HMM
from .bayesmodel import BayesianParameter
from ..priors import DirichletPrior
from ..utils import logsumexp


class PhoneLoop(HMM):
    'Phone Loop HMM.'

    @classmethod
    def create(cls, graph, start_pdf, end_pdf, modelset, weights=None,
               prior_strength=1.0):
        'Create a :any:`PhoneLoop` model.'
        mf_groups = modelset.mean_field_factorization()
        prior_nparams = mf_groups[0][0].prior.natural_parameters
        dtype, device = prior_nparams.dtype, prior_nparams.device

        if weights is None:
            weights = torch.ones(len(start_pdf), dtype=dtype, device=device)
            weights /= len(start_pdf)
        else:
            weights = torch.tensor(weights, dtype=dtype, device=device,
                                   requires_grad=False)
        prior_weights = DirichletPrior(prior_strength * weights)
        posterior_weights = DirichletPrior(prior_strength * weights)
        return cls(graph, modelset, start_pdf, end_pdf, prior_weights,
                   posterior_weights)

    def __init__(self, graph, modelset, start_pdf, end_pdf, prior_weights,
                 posterior_weights):
        super().__init__(graph, modelset)
        self.start_pdf = start_pdf
        self.end_pdf = end_pdf
        self.weights = BayesianParameter(prior_weights, posterior_weights)
        self.weights.register_callback(self._on_weights_update)
        self._on_weights_update()

    def _on_weights_update(self):
        log_weights = self.weights.expected_natural_parameters()
        start_idxs = [value for value in self.start_pdf.values()]
        for end_idx in self.end_pdf.values():
            self.graph.value.trans_log_probs[end_idx, start_idxs] = log_weights

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        params = super().mean_field_factorization()
        params[0] += [self.weights]
        return params

    def accumulate(self, stats, parent_msg=None):
        retval = super().accumulate(stats, parent_msg)
        dtype = self.cache['resps'].dtype
        device = self.cache['resps'].device

        # Re-map the responsibilities.
        nstates = self.graph.value.n_states
        trans_resps = torch.zeros(nstates, nstates, dtype=dtype, device=device)
        mapping = self.cache['inference_graph'].pdf_id_mapping
        tmp = trans_resps[mapping, :]
        tmp[:, mapping] = self.cache['trans_resps'].sum(dim=0)
        trans_resps[mapping, :] = tmp
        resps = torch.zeros(len(self.cache['resps']), nstates, dtype=dtype,
                            device=device)
        resps[:, mapping] = self.cache['resps']

        start_idxs = [value for value in self.start_pdf.values()]
        end_idxs = [value for value in self.end_pdf.values()]
        phone_resps = trans_resps[:, start_idxs]
        phone_resps = phone_resps[end_idxs, :].sum(dim=0)
        phone_resps += resps[0][start_idxs]
        retval.update({self.weights: phone_resps})
        return retval


__all__ = ['PhoneLoop']

