import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily


__all__ = ['IsotropicNormalGamma', 'IsotropicNormalGammaStdParams',
           'JointIsotropicNormalGamma', 'JointIsotropicNormalGammaStdParams']


@dataclass(init=False, eq=False, unsafe_hash=True)
class IsotropicNormalGammaStdParams(torch.nn.Module):
    'Standard parameterization of the Normal-Gamma pdf.'

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
        dim = len(natural_params) - 3
        np1 = natural_params[:dim]
        np2 = natural_params[dim:dim+1]
        np3 = natural_params[-2].view(1)
        np4 = natural_params[-1].view(1)
        scale = -2 * np3
        shape = np4 + 1 - .5 * dim
        mean = np1 / scale
        rate = -np2 - .5 * scale * torch.sum(mean * mean, dim=-1)
        return cls(mean, scale, shape, rate)


class IsotropicNormalGamma(ExponentialFamily):
    '''Set of independent Normal-Gamma distribution having the same
    scale (Normal), shape and rate (Gamma) parameters for all dimension.

    '''

    _std_params_def = {
        'mean': 'Mean of the Normal.',
        'scale': 'Scale of the (diagonal) covariance matrix.',
        'shape': 'Shape parameter of the Gamma (shared across dimension).',
        'rate': 'Rate parameter of the Gamma (shared across dimension).'
    }

    @property
    def dim(self):
        '''Return a tuple with the dimension of the Normal and the
        dimension of the joint Gamma (just one for the latter).

        '''
        return (len(self.params.mean), 1)

    @property
    def conjugate_sufficient_statistics_dim(self):
        return len(self.params.mean) + 1

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
        dim = self.dim[0]
        precision = self.params.shape / self.params.rate
        logdet = (torch.digamma(self.params.shape) - torch.log(self.params.rate))
        return torch.cat([
            precision * self.params.mean,
            precision.view(1),
            ((dim / self.params.scale) \
                    + precision * (self.params.mean**2).sum()).view(1),
            logdet.view(1)
        ])

    def expected_value(self):
        'Expected mean and expected precision.'
        return self.params.mean, self.params.shape / self.params.rate

    def log_norm(self):
        dim = self.dim[0]
        return torch.lgamma(self.params.shape) \
            - self.params.shape * self.params.rate.log() \
            - .5 * dim * self.params.scale.log()

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (m=mean, k=scale, a=shape, b=rate) the
        natural parameterization is given by:

        nparams = (
            k * m ,
            -.5 * k * m^2
            -.5 * k,
            a - 1 + .5 * D
        )

        Note:
            "D" is the dimension of "m" and "^2" is the elementwise
            square operation.

        Returns:
            ``torch.Tensor[D + 3]``

        '''
        return torch.cat([
            self.params.scale * self.params.mean,
            (-.5 * self.params.scale * torch.sum(self.params.mean**2) \
                    - self.params.rate).view(1),
            -.5 * self.params.scale.view(1),
            self.params.shape.view(1) - 1 + .5 * self.dim[0]
        ])

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)

    def sufficient_statistics_from_rvectors(self, rvecs):
        '''
        Real vector z = (x, y)
        \mu = x
        \sigma^2 = \exp(y)

        '''
        dim = self.dim[0]
        mean = rvecs[:, :dim]
        log_precision = rvecs[:, -2:-1]
        precision = torch.exp(log_precision)
        return torch.cat([
            precision * mean,
            precision,
            torch.sum(precision * (mean ** 2), dim=-1)[:, None],
            torch.sum(log_precision, dim=-1)[:, None]
        ], dim=-1)


@dataclass(init=False, eq=False, unsafe_hash=True)
class JointIsotropicNormalGammaStdParams(torch.nn.Module):
    means: torch.Tensor
    scales: torch.Tensor
    shape: torch.Tensor
    rate: torch.Tensor

    def __init__(self, means, scales, shape, rate):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('scales', scales)
        self.register_buffer('shape', shape)
        self.register_buffer('rate', rate)

    @classmethod
    def from_natural_parameters(cls, natural_params, ncomp):
        dim = (len(natural_params) - 2 - ncomp) // ncomp
        np1s = natural_params[:ncomp * dim].reshape(ncomp, dim)
        np2 = natural_params[ncomp * dim: ncomp * dim + 1]
        np3s = natural_params[-(ncomp + 1):-1]
        np4 = natural_params[-1]
        scales = -2 * np3s
        shape = np4 + 1 - .5 * dim * ncomp
        means = np1s / scales[:, None]
        rate = -np2 - .5 * (scales * (means * means).sum(dim=-1)).sum()
        return cls(means, scales, shape, rate)


class JointIsotropicNormalGamma(ExponentialFamily):
    '''Set of Normal distributions sharing the same gamma prior over
    the precision.

    '''

    _std_params_def = {
        'means': 'Set of mean parameters.',
        'scales': 'Set of scaling of the precision (for each Normal).',
        'shape': 'Shape parameter (Gamma).',
        'rate': 'Rate parameter (Gamma).'
    }

    @property
    def dim(self):
        '''Return a tuple ((k, D), 1)' where K is the number of Normal
        and D is the dimension of their support.

        '''
        return (tuple(self.params.means.shape), 1)

    @property
    def conjugate_sufficient_statistics_dim(self):
        dim = self.dim
        return dim[0][0] * dim[0][1] + 1

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variables mu (set of vector), l (positive scalar)
        the sufficient statistics of the joint isotropic Normal-Gamma
        are given by:

        stats = (
            l_1 * mu_1,
            ...,
            l_k * mu_k
            l,
            l * mu^2_i,
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

        Note: ""D" is the dimenion of "m", "k" is the number of Normal,
            and "psi" is the "digamma" function.

.       '''
        dim = self.dim[0][1]
        precision = self.params.shape / self.params.rate
        logdet = torch.digamma(self.params.shape) - torch.log(self.params.rate)
        return torch.cat([
            precision * self.params.means.reshape(-1),
            precision.view(1),
            ((dim / self.params.scales) \
             + precision * (self.params.means**2).sum(dim=-1)).reshape(-1),
            logdet.view(1)
        ])

    def expected_value(self):
        'Expected means and expected precision (scalar).'
        return self.params.means, self.params.shape / self.params.rate

    def log_norm(self, natural_parameters=None):
        dim = self.dim[0][1]
        return torch.lgamma(self.params.shape) \
            - self.params.shape * self.params.rate.log() \
            - .5 * dim * self.params.scales.log().sum()

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        dim = self.dim[0][1]
        ncomp = self.dim[0][0]
        return torch.cat([
            (self.params.scales[:, None] * self.params.means).reshape(-1),
            (-.5 * (self.params.scales * (self.params.means**2).sum(dim=-1)).sum() \
             - self.params.rate).view(1),
            -.5 * self.params.scales.view(-1),
            self.params.shape.view(1) - 1 + .5 * dim * ncomp,
        ])

    def update_from_natural_parameters(self, natural_params):
        ncomp = self.dim[0][0]
        self.params = self.params.from_natural_parameters(natural_params, ncomp)

    def sufficient_statistics_from_rvectors(self, rvecs):
        '''
        Real vector z = (x, y)
        \mu = x
        \sigma^2 = \exp(y)

        '''
        k, dim = self.dim[0]
        means = rvecs[:, :k * dim].reshape(-1, k, dim)
        log_precision = rvecs[:, -2:-1]
        precision = torch.exp(log_precision)
        return torch.cat([
            (means * precision[:, None, :]).reshape(-1, k * dim),
            precision,
            torch.sum((means ** 2) * precision[:, None, :], dim=-1),
            torch.sum(log_precision, dim=-1)[:, None]
        ], dim=-1)
