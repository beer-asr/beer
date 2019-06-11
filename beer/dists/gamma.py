import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihood


__all__ = ['GammaLikelihood', 'Gamma', 'GammaStdParams']

EPS = 1e-6

@dataclass
class GammaLikelihood(ConjugateLikelihood):
    dim: int

    def sufficient_statistics_dim(self, zero_stats=True):
        zero_stats_dim = 1 if zero_stats else 0
        return 2 * self.dim + zero_stats_dim

    @staticmethod
    def sufficient_statistics(data):
        dim, dtype, device = data.shape[1], data.dtype, data.device
        return torch.cat([
            -data,
            data.log(),
            torch.ones(len(data), 1, dtype=data.dtype, device=data.device)
        ], dim=-1)

    def parameters_from_pdfvector(self, pdfvec):
        size = pdfvec.shape
        if len(size) == 1:
            pdfvec = pdfvec.view(1, -1)
        dim = self.dim
        rate = -pdfvec[:, :dim]
        shape = pdfvec[:, dim: 2 * dim] + 1
        if len(size) == 1:
            return shape.view(-1), rate.view(-1)
        return shape.view(-1, dim), rate.view(-1, dim)


    def pdfvectors_from_rvectors(self, rvecs):
        dim = rvecs.shape[-1] // 2
        shape = (EPS + rvecs[:, :dim]).exp()
        rate = (EPS + rvecs[:, dim:]).exp()
        lnorm = torch.lgamma(shape) - shape * torch.log(rate)
        lnorm = torch.sum(lnorm, dim=-1, keepdim=True)
        retval =  torch.cat([-rate, shape - 1, -lnorm], dim=-1)
        return retval

    def __call__(self, pdfvecs, stats):
        if len(pdfvecs.shape) == 1:
            pdfvecs = pdfvecs.view(1, -1)
        return stats @ pdfvecs.t()


@dataclass(init=False, unsafe_hash=True)
class GammaStdParams(torch.nn.Module):
    shape: torch.Tensor
    rate: torch.Tensor

    def __init__(self, shape, rate):
        super().__init__()
        self.register_buffer('shape', shape)
        self.register_buffer('rate', rate)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        npsize = natural_params.shape
        if len(npsize) == 1:
            natural_params = natural_params.view(1, -1)
        dim = natural_params.shape[-1] // 2
        shape = natural_params[:, dim:] + 1
        rate = -natural_params[:, :dim]
        if len(npsize) == 1:
            return cls(shape.view(-1), rate.view(-1))
        return cls(shape, rate)


class Gamma(ExponentialFamily):
    _std_params_def = {
        'shape': 'Shape parameter of the Gamma.',
        'rate': 'Rate parameter of the Gamma.'
    }

    _std_params_cls = GammaStdParams

    def __len__(self):
        paramshape = self.params.mean.shape
        return 1 if len(paramshape) <= 1 else paramshape[0]

    @property
    def dim(self):
        shape = self.params.shape
        size = len(shape.shape) if len(shape.shape) > 0 else 1
        if size == 1:
            return len(shape)
        return tuple(shape.shape)

    def conjugate(self):
        return GammaLikelihood(self.params.shape.shape[-1])

    def expected_sufficient_statistics(self):
        '''
        stats = (
            x,
            ln(x)
        )

        E[stats] = (
            a / b,
            psi(a) - ln(b),
        )
        '''
        shape, rate = self.params.shape, self.params.rate
        return torch.cat([shape / rate, torch.digamma(shape) - torch.log(rate)],
                         dim=-1)

    def expected_value(self):
        return self.params.shape / self.params.rate

    def log_norm(self):
        shape, rate = self.params.shape, self.params.rate
        return  (torch.lgamma(shape) - shape * torch.log(rate)).sum(dim=-1)

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''
        nparams = (
            k * m ,
            -.5 * (k * m^2 - b)
            -.5 * k,
            a - .5
        )
        '''
        shape, rate = self.params.shape, self.params.rate
        return torch.cat([-rate, shape - 1], dim=-1)

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)
