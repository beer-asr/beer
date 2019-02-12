
import torch
from .hmm import HMM
from .parameters import BayesianParameter
from ..dists import Dirichlet, DirichletStdParams
from ..utils import logsumexp


class PhoneLoop(HMM):
    'Phone Loop HMM.'

    @classmethod
    def create(cls, graph, start_pdf, end_pdf, modelset, weights=None,
               prior_strength=1.0):
        'Create a :any:`PhoneLoop` model.'
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
        params = DirichletStdParams(prior_strength * weights)
        prior_weights = Dirichlet(params)
        params = DirichletStdParams(prior_strength * weights)
        posterior_weights = Dirichlet(params)
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
            self.graph.trans_log_probs[end_idx, start_idxs] = log_weights

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        params = super().mean_field_factorization()
        params[0] += [self.weights]
        return params

    def accumulate(self, stats, parent_msg=None):
        retval = super().accumulate(stats, parent_msg)
        trans_resps = self.cache['trans_resps'].sum(dim=0)
        start_idxs = [value for value in self.start_pdf.values()]
        end_idxs = [value for value in self.end_pdf.values()]
        phone_resps = trans_resps[:, start_idxs]
        phone_resps = phone_resps[end_idxs, :].sum(dim=0)
        phone_resps += self.cache['resps'][0][start_idxs]
        retval.update({self.weights: phone_resps})
        return retval


__all__ = ['PhoneLoop']

