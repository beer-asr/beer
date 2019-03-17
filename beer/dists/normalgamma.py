import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihood


__all__ = ['NormalDiagonalLikelihood', 'NormalGamma', 'NormalGammaStdParams']

@dataclass
class NormalDiagonalLikelihood(ConjugateLikelihood):
    dim: int

    def sufficient_statistics_dim(self, zero_stats=True):
        zero_stats_dim = 2 if zero_stats else 0
        return 2 * self.dim + zero_stats_dim

    @staticmethod
    def sufficient_statistics(data):
        dim, dtype, device = data.shape[1], data.dtype, data.device
        return torch.cat([
            data,
            -.5 * data**2,
            -.5 * torch.ones(len(data), 1, dtype=dtype, device=device),
            .5 * torch.ones(len(data), 1, dtype=dtype, device=device),
        ], dim=-1)

    def parameters_from_pdfvector(self, pdfvec):
        size = pdfvec.shape
        if len(size) == 1:
            pdfvec = pdfvec.view(1, -1)
        dim = self.dim
        precision = pdfvec[:, dim: 2 * dim]
        mean = pdfvec[:, :dim] / precision
        if len(size) == 1:
            return mean.view(-1), precision.view(-1)
        return mean.view(-1, dim), precision.view(-1, dim)


    def pdfvectors_from_rvectors(self, rvecs):
        dim = rvecs.shape[-1] // 2
        mean = rvecs[:, :dim]
        log_precision = rvecs[:, dim: 2 * dim]
        precision = torch.exp(log_precision)
        log_basemeasure = self.dim * math.log(2 * math.pi)
        retval =  torch.cat([
            precision * mean,
            precision,
            torch.sum(precision * (mean ** 2), dim=-1)[:, None],
            torch.sum(log_precision, dim=-1)[:, None]
        ], dim=-1)
        return retval

    def __call__(self, pdfvecs, stats):
        if len(pdfvecs.shape) == 1:
            pdfvecs = pdfvecs.view(1, -1)
        log_basemeasure = -.5 * self.dim * math.log(2 * math.pi)
        return stats @ pdfvecs.t() + log_basemeasure


@dataclass(init=False, unsafe_hash=True)
class NormalGammaStdParams(torch.nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor
    rates: torch.Tensor

    def __init__(self, mean, scale, shape, rates):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('shape', shape)
        self.register_buffer('rates', rates)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        npsize = natural_params.shape
        if len(npsize) == 1:
            natural_params = natural_params.view(1, -1)
        dim = (natural_params.shape[-1] -  2) // 2
        np1 = natural_params[:, :dim]
        np2 = natural_params[:, dim:2*dim]
        np3 = natural_params[:, -2]
        np4 = natural_params[:, -1]
        scale = -2 * np3
        shape = np4 + .5
        mean = np1 / scale[:, None]
        rates = -np2 - .5 * scale[:, None] * mean**2

        if len(npsize) == 1:
            return cls(mean.view(-1), scale.view(-1), shape.view(-1),
                       rates.view(-1))
        return cls(mean, scale.view(-1, 1), shape.view(-1, 1), rates)


class NormalGamma(ExponentialFamily):
    _std_params_def = {
        'mean': 'Mean of the Normal.',
        'scale': 'Scale of the (diagonal) covariance matrix.',
        'shape': 'Shape parameter of the Gamma (shared across dimension).',
        'rates': 'Rate parameters of the Gamma.'
    }

    _std_params_cls = NormalGammaStdParams

    def __len__(self):
        paramshape = self.params.mean.shape
        return 1 if len(paramshape) <= 1 else paramshape[0]

    @property
    def dim(self):
        return (*self.params.mean.shape, self.params.rates.shape[-1])

    def conjugate(self):
        return NormalDiagonalLikelihood(self.params.mean.shape[-1])

    def expected_sufficient_statistics(self):
        '''
        stats = (
            l * mu,
            l,
            \sum_i (l * mu^2)_i,
            \sum_i ln l_i
        )

        E[stats] = (
            (a / b) * m,
            (a / b),
            (D/k) + \sum_i ((a / b) * m^2)_i,
            \sum_i psi(a) - ln(b_i)
        )
        '''
        mean, scale, shape, rates = self.params.mean, self.params.scale, \
            self.params.shape, self.params.rates
        dim = mean.shape[-1]
        shape_size = shape.shape if len(shape.shape) > 0 else 1
        diag_precision = shape / rates
        prec_quad_mean = (diag_precision * mean**2).sum(dim=-1, keepdim=True)
        prec_quad_mean += (dim / scale)
        logdet = torch.sum(torch.digamma(shape) - torch.log(rates), dim=-1)
        return torch.cat([
            diag_precision * mean, diag_precision,
            prec_quad_mean.reshape(shape_size),
            logdet.reshape(shape_size)],
        dim=-1)

    def expected_value(self):
        return self.params.mean, self.params.shape / self.params.rates

    def log_norm(self):
        scale, shape, rates = self.params.scale, self.params.shape, \
            self.params.rates
        dim = rates.shape[-1]
        return (dim * torch.lgamma(shape) \
            - shape * rates.log().sum(dim=-1, keepdim=True) \
            - .5 * dim * scale.log()).sum(dim=-1)

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
        mean, scale, shape, rates = self.params.mean, self.params.scale, \
            self.params.shape, self.params.rates
        shape_size = shape.shape if len(shape.shape) > 0 else 1
        return torch.cat([
            scale * mean,
            -.5 * scale * mean**2 - rates,
            -.5 * scale.view(shape_size),
            shape.view(shape_size) - .5,
        ], dim=-1)

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)
