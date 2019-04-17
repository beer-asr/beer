import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihood


__all__ = ['IsotropicNormalGamma', 'IsotropicNormalGammaStdParams',
           'IsotropicNormalLikelihood']

@dataclass
class IsotropicNormalLikelihood(ConjugateLikelihood):
    dim: int

    def sufficient_statistics_dim(self, zero_stats=True):
        zero_stats_dim = 2 if zero_stats else 0
        return self.dim + 1 + zero_stats_dim

    @staticmethod
    def sufficient_statistics(data):
        dim = data.shape[-1]
        length = len(data)
        tensorconf = {'dtype': data.dtype, 'device': data.device}
        return torch.cat([
            data,
            -.5 * torch.sum(data**2, dim=-1).view(-1, 1),
            -.5 * torch.ones(length, 1, **tensorconf),
            .5 * dim * torch.ones(length, 1, **tensorconf),
        ], dim=-1)

    @staticmethod
    def pdfvectors_from_rvectors(rvecs):
        dim = rvecs.shape[-1] - 1
        mean = rvecs[:, :dim]
        log_precision = rvecs[:, -2:-1]
        precision = torch.exp(log_precision)
        return torch.cat([
            precision * mean,
            precision,
            torch.sum(precision * (mean ** 2), dim=-1)[:, None],
            torch.sum(log_precision, dim=-1)[:, None]
        ], dim=-1)

    @staticmethod
    def parameters_from_pdfvector(pdfvec):
        size = pdfvec.shape
        if len(size) == 1:
            pdfvec = pdfvec.view(1, -1)
        precision = pdfvec[:, -3]
        mean = pdfvec[:, :-3] / precision
        if len(size) == 1:
            return mean.view(-1), precision.view(-1)
        return mean.view(-1, dim), precision.view(-1, dim)

    def __call__(self, pdfvecs, stats):
        if len(pdfvecs.shape) == 1:
            pdfvecs = pdfvecs.view(1, -1)
        log_basemeasure = -.5 * self.dim * math.log(2 * math.pi)
        return stats @ pdfvecs.t() + log_basemeasure


@dataclass(init=False, unsafe_hash=True)
class IsotropicNormalGammaStdParams(torch.nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor
    rate: torch.Tensor

    def __init__(self, mean, scale, shape, rate):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('shape', shape)
        self.register_buffer('rate', rate)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        npsize = natural_params.shape
        if len(npsize) == 1:
            natural_params = natural_params.view(1, -1)
        dim = natural_params.shape[-1] - 3
        np1 = natural_params[:, :dim]
        np2 = natural_params[:, dim:dim+1]
        np3 = natural_params[:, -2].view(-1, 1)
        np4 = natural_params[:, -1].view(-1, 1)
        scale = -2 * np3
        shape = np4 + 1 - .5 * dim
        mean = np1 / scale
        rate = -np2 - .5 * scale * torch.sum(mean * mean, dim=-1, keepdim=True)

        if len(npsize) == 1:
            return cls(mean.view(-1), scale.view(-1), shape.view(-1),
                       rate.view(-1))
        return cls(mean, scale.view(-1, 1), shape.view(-1, 1), rate.view(-1, 1))


class IsotropicNormalGamma(ExponentialFamily):
    _std_params_def = {
        'mean': 'Mean of the Normal.',
        'scale': 'Scale of the (diagonal) covariance matrix.',
        'shape': 'Shape parameter of the Gamma (shared across dimension).',
        'rate': 'Rate parameter of the Gamma (shared across dimension).'
    }

    _std_params_cls = IsotropicNormalGammaStdParams

    def __len__(self):
        paramshape = self.params.mean.shape
        return 1 if len(paramshape) <= 1 else paramshape[0]

    @property
    def dim(self):
        return (*self.params.mean.shape, 1)

    def conjugate(self):
        return IsotropicNormalLikelihood(self.params.mean.shape[-1])

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable mu (vector), l (positive scalar)
        the sufficient statistics of the isotropic Normal-Gamma are
        given by:

        stats = (
            l * mu,
            l,
            l * \sum_i mu^2_i,
            D * ln(l)
        )

        For the standard parameters (m=mean, k=scale, a=shape, b=rate)
        expectation of the sufficient statistics is given by:

        E[stats] = (
            (a / b) * m,
            (a / b),
            (D/k) + (a / b) * \sum_i m^2_i,
            (psi(a) - ln(b))
        )

        Note: ""D" is the dimenion of "m"
            and "psi" is the "digamma" function.

        '''
        mean, scale, shape, rate = self.params.mean, self.params.scale, \
            self.params.shape, self.params.rate
        dim = mean.shape[-1]
        shape_size = shape.shape if len(shape.shape) > 0 else 1
        precision = shape / rate
        prec_quad_mean = (precision * (mean**2).sum(dim=-1, keepdim=True))
        prec_quad_mean += (dim / scale)
        logdet = (torch.digamma(shape) - torch.log(rate))
        return torch.cat([precision * mean, precision.reshape(shape_size),
                          prec_quad_mean, logdet.reshape(shape_size)], dim=-1)

    def expected_value(self):
        'Expected mean and expected precision.'
        return self.params.mean, self.params.shape / self.params.rate

    def log_norm(self):
        mean, scale, shape, rate = self.params.mean, self.params.scale, \
            self.params.shape, self.params.rate
        dim = mean.shape[-1]
        return (torch.lgamma(shape) - shape * rate.log() \
                - .5 * dim * scale.log()).sum(dim=-1)

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''
        nparams = (
            k * m ,
            -.5 * k * m^2
            -.5 * k,
            a - 1 + .5 * D
        )
        '''
        mean, scale, shape, rate = self.params.mean, self.params.scale, \
            self.params.shape, self.params.rate
        dim = mean.shape[-1]
        shape_size = shape.shape if len(shape.shape) > 0 else 1
        return torch.cat([
            scale * mean,
            -.5 * scale * torch.sum(mean**2, dim=-1, keepdim=True) - rate,
            -.5 * scale.view(shape_size),
            shape.view(shape_size) - 1 + .5 * dim
        ], dim=-1)

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)
