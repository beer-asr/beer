from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihood


__all__ = ['CategoricalLikelihood', 'Dirichlet', 'DirichletStdParams']


@dataclass
class CategoricalLikelihood(ConjugateLikelihood):
    dim: int

    def sufficient_statistics_dim(self, zero_stats=True):
        zero_stats_dim = 1 if zero_stats else 0
        return self.dim - 1 + zero_stats_dim

    def sufficient_statistics(self, data):
        retval = data.clone()
        retval[:, -1] = retval.sum(dim=-1)
        return retval

    @staticmethod
    def log_norm(nparams):
        # Stable implementation of the log-normalizer of a categorical
        # equivalent of the naive implementation:
        #   lognorm = torch.log(1 + rvectors.exp())
        return torch.nn.functional.softplus(torch.logsumexp(nparams, dim=-1))

    def parameters_from_pdfvector(self, pdfvec):
        size = pdfvec.shape
        if len(size) == 1:
            pdfvec = pdfvec.view(1, -1)
        retval = torch.zeros_like(pdfvec, requires_grad=False)
        lnorm = CategoricalLikelihood.log_norm(pdfvec[:, :-1])
        remainder = (-lnorm).exp()
        retval[:, :-1] = pdfvec[:, :-1].exp() * remainder[:, None]
        retval[:, -1] = remainder
        if len(size) == 1:
            return retval.view(-1)
        return retval.view(-1, self.dim)

    @staticmethod
    def pdfvectors_from_rvectors(rvecs):
        '''
        Returns:

            (x, -A(x))

        with:
             A(x) = ln( 1 + exp( sum_i^{D-1} (x_i) ) )

        '''
        lnorm = CategoricalLikelihood.log_norm(rvecs).view(-1, 1)
        return torch.cat([rvecs, -lnorm], dim=-1)

    def __call__(self, pdfvecs, stats):
        size = len(pdfvecs.shape)
        return stats @ pdfvecs.t() if size > 1 else stats @ pdfvecs


@dataclass(init=False, unsafe_hash=True)
class DirichletStdParams(torch.nn.Module):
    concentrations: torch.Tensor

    def __init__(self, concentrations):
        super().__init__()
        self.register_buffer('concentrations', concentrations)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        npsize = natural_params.shape
        if len(npsize) == 1:
            natural_params = natural_params.view(1, -1)
        concentrations = natural_params + 1
        concentrations[:, -1] = natural_params[:, -1] \
                                - concentrations[:, :-1].sum(dim=-1) + 1

        if len(npsize) == 1:
            return cls(concentrations.view(-1))
        return cls(concentrations)


class Dirichlet(ExponentialFamily):
    _std_params_def = {
        'concentrations': 'Concentrations parameter.'
    }

    _std_params_cls = DirichletStdParams

    def __len__(self):
        paramshape = self.params.concentrations.shape
        return 1 if len(paramshape) <= 1 else paramshape[0]

    def conjugate(self):
        return CategoricalLikelihood(self.params.concentrations.shape[-1])

    @property
    def dim(self):
        concentrations = self.params.concentrations
        size = len(concentrations.shape) if len(concentrations.shape) > 0 else 1
        if size == 1:
            return len(concentrations)
        return tuple(concentrations.shape)

    def expected_sufficient_statistics(self):
        '''
        stats = (
            ln(p_i / (1 - sum_j p_j))
            ln( 1 - sum_j p_j )
        )

        E[stats] = (
            psi(a_i) - pis(a_d)
            psi(a_d) - psi( sum_i^{d-1} a_i)
        )
        '''
        concentrations = self.params.concentrations
        size = len(concentrations.shape) if len(concentrations.shape) > 0 else 1
        if size == 1:
            concentrations = concentrations.view(1, -1)
        psis = torch.digamma(concentrations)
        psi_sum = torch.digamma(concentrations.sum(dim=-1))
        retval = psis - psis[:, -1].view(-1, 1)
        retval[:, -1] = psis[:, -1] - psi_sum
        if size == 1:
            return retval.view(-1)
        return retval

    def expected_value(self):
        'Expected distribution p.'
        norm = self.params.concentrations.sum(dim=-1, keepdim=True)
        return self.params.concentrations / norm

    def log_norm(self):
        concentrations = self.params.concentrations
        return torch.lgamma(concentrations).sum(dim=-1) \
               - torch.lgamma(concentrations.sum(dim=-1))

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''
        nparams = (
            a_i - 1,
            sum_i a_i + a_d - 1
        )
        '''
        concentrations = self.params.concentrations
        size = len(concentrations.shape) if len(concentrations.shape) > 0 else 1
        if size == 1:
            concentrations = concentrations.view(1, -1)
        retval = concentrations - 1
        retval[:, -1] = concentrations.sum(dim=-1) - 1
        if size == 1:
            return retval.view(-1)
        return retval

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)
