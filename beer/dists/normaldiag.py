import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihoodDescriptor


__all__ = ['NormalDiagonalCovariance', 'NormalDiagonalCovarianceStdParams',
           'JointNormalDiagonalCovariance', 
           'JointNormalDiagonalCovarianceStdParams',
           'NormalFixedDiagonalCovarianceLikelihood',
           'JointNormalFixedDiagonalCovarianceLikelihood',]


class NormalFixedDiagonalCovarianceLikelihood(ConjugateLikelihoodDescriptor):

    __slots__ = 'dim'

    def __init__(self, dim):
        self.dim = dim

    def __repr__(self):
        return f'{self.__class__.__qualname__}(dim={self.dim})'

    @property
    def sufficient_statistics_dim(self):
        return self.dim 

    def parameters_from_pdfvector(self, pdfvec):
        return pdfvec[:self.dim]

    def pdfvectors_from_rvectors(self, rvecs):
        return torch.cat([rvecs, rvecs**2], dim=-1)


@dataclass(init=False, eq=False, unsafe_hash=True)
class NormalDiagonalCovarianceStdParams(torch.nn.Module):
    '''Standard parameterization of the Normal pdf with diagonal 
    covariance matrix.
    '''

    mean: torch.Tensor
    diag_cov: torch.Tensor

    def __init__(self, mean, diag_cov):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('diag_cov', diag_cov)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        dim = (len(natural_params)) // 2
        np1 = natural_params[:dim]
        np2 = natural_params[dim:2 * dim]
        diag_cov = 1. / (-2 * np2)
        mean = diag_cov * np1
        return cls(mean, diag_cov)


class NormalDiagonalCovariance(ExponentialFamily):
    'Normal pdf with diagonal covariance matrix.'

    _std_params_def = {
        'mean': 'Mean parameter.',
        'diagona_cov': 'Diagonal of the covariance matrix.',
    }

    @property
    def dim(self):
        return len(self.params.mean)

    def conjugate(self):
        return NormalFixedDiagonalCovarianceLikelihood(self.dim)

    def forward(self, data):
        nparams = self.natural_parameters()
        stats = torch.cat([data, data**2], dim=-1)
        return stats @ nparams - .5 * self.dim * math.log(2 * math.pi)

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable x (vector)the sufficient statistics of 
        the Normal with diagonal covariance matrix are given by:

        stats = (
            x,
            x**2,
        )

        For the standard parameters (m=mean, s=diagonal of the cov. 
        matrix) the expectation of the sufficient statistics is
        given by:

        E[stats] = (
            m,
            s + m**2
        )

        '''
        return torch.cat([
            self.params.mean,
            self.params.diag_cov + self.params.mean ** 2
        ])

    def expected_value(self):
        return self.params.mean

    def log_norm(self):
        dim = self.dim
        mean = self.params.mean
        diag_prec = 1./ self.params.diag_cov
        log_base_measure = -.5 * dim * math.log(2 * math.pi)
        return -.5 * (diag_prec * mean) @ mean \
                + .5 * diag_prec.log().sum() \
                + log_base_measure

    def sample(self, nsamples):
        mean = self.params.mean
        diag_cov = self.params.diag_cov
        noise = torch.randn(nsamples, self.dim, dtype=mean.dtype, 
                            device=mean.device)
        return mean[None] + diag_cov[None] * noise

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (m=mean, s=diagonal of the cov. matrix) the
        natural parameterization is given by:

        nparams = (
            s^-1 * m ,
            -.5 * s^1
        )

        Returns:
            ``torch.Tensor[2 * D]``

        '''
        mean = self.params.mean
        diag_prec = 1. / self.params.diag_cov
        return torch.cat([diag_prec * mean, -.5 * diag_prec])

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)


class JointNormalFixedDiagonalCovarianceLikelihood(ConjugateLikelihoodDescriptor):

    __slots__ = 'ncomp', 'dim'

    def __init__(self, ncomp, dim):
        self.ncomp = ncomp
        self.dim = dim

    def __repr__(self):
        return f'{self.__class__.__qualname__}(ncomp={self.ncomp}, dim={self.dim})'

    @property
    def sufficient_statistics_dim(self):
        return self.ncomp * self.dim + self.dim

    def parameters_from_pdfvector(self, pdfvec):
        ndim = self.ncomp * self.dim
        return pdfvec[:ndim].reshape(self.ncomp, self.dim)
    
    def pdfvectors_from_rvectors(self, rvecs):
        k, dim = self.ncomp, self.dim
        means = rvecs[:, :k * dim].reshape(-1, k, dim)
        log_precision = rvecs[:, k * dim:]
        precision = torch.exp(log_precision)
        return torch.cat([
            (means * precision[:, None, :]).reshape(-1, k * dim),
            precision,
            torch.sum((means ** 2) * precision[:, None, :], dim=-1),
            torch.sum(log_precision, dim=-1)[:, None]
        ], dim=-1)

    def pdfvectors_from_rvectors(self, rvecs):
        return torch.cat([rvecs, rvecs**2], dim=-1)


@dataclass(init=False, eq=False, unsafe_hash=True)
class JointNormalDiagonalCovarianceStdParams(torch.nn.Module):
    means: torch.Tensor
    diag_covs: torch.Tensor

    def __init__(self, means, diag_covs):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('diag_covs', diag_covs)

    @classmethod
    def from_natural_parameters(cls, natural_params, ncomp):
        dim = (len(natural_params)) // (2 * ncomp)
        np1s = natural_params[:ncomp * dim]
        np2s = natural_params[ncomp * dim:2 * ncomp * dim]
        diag_covs = 1. / (-2 * np2s)
        means = diag_covs * np1s
        return cls(means.reshape(ncomp, dim), diag_covs.reshape(ncomp, dim))
        

class JointNormalDiagonalCovariance(ExponentialFamily):
    '''Set of Normal distributions sharing the same Gamma prior over
    the diagonal of the precision matrix.

    '''

    _std_params_def = {
        'means': 'Set of mean parameters.',
        'diag_covs': 'Set of diagonal covariance matrices.'
    }

    @property
    def dim(self):
        '''Return a tuple ((K, D))' where K is the number of Normal
        and D is the dimension of their support.

        '''
        return tuple(self.params.means.shape)

    def conjugate(self):
        return JointNormalFixedDiagonalCovarianceLikelihood(**self.dim)

    def forward(self, data):
        means = self.params.means
        diag_precs = 1. / self.params.diag_covs
        nparams = torch.cat([diag_precs * means, -.5 * diag_precs], dim=-1)
        stats = torch.cat([data, data**2], dim=-1)
        return torch.sum(stats[:, :] * nparams[:, None, :], dim=-1) \
            - .5 * self.dim[1] * math.log(2 * math.pi)

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable x (vector)the sufficient statistics of 
        the Normal with diagonal covariance matrix are given by:

        stats = (
            x,
            x**2,
        )

        For the standard parameters (m=mean, s=diagonal of the cov. 
        matrix) the expectation of the sufficient statistics is
        given by:

        E[stats] = (
            m,
            s + m**2
        )

        '''
        means = self.params.means
        return torch.cat([
            means.reshape(-1),
            (self.params.diag_covs + means**2).reshape(-1)
        ])

    def expected_value(self):
        'Expected means and expected diagonal of the precision matrix.'
        return self.params.means

    def log_norm(self):
        dim = self.dim[0] * self.dim[1]
        mean = self.params.means.reshape(-1)
        diag_prec = 1./ self.params.diag_covs.reshape(-1)
        log_base_measure = -.5 * dim * math.log(2 * math.pi)
        return -.5 * (diag_prec * mean) @ mean \
                + .5 * diag_prec.log().sum() \
                + log_base_measure
        
    def sample(self, nsamples):
        means = self.params.means
        diag_covs = self.params.diag_covs
        noise = torch.randn(nsamples, *self.dim, dtype=means.dtype, 
                            device=means.device)
        samples = means[None] + diag_covs[None] * noise
        return samples.permute(1, 0, 2)

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (m=mean, s=diagonal of the cov. matrix) the
        natural parameterization is given by:

        nparams = (
            s^-1 * m ,
            -.5 * s^1
        )

        Returns:
            ``torch.Tensor[2 * D]``

        '''
        mean = self.params.means.reshape(-1)
        diag_prec = 1. / self.params.diag_covs.reshape(-1)
        return torch.cat([diag_prec * mean, -.5 * diag_prec])

    def update_from_natural_parameters(self, natural_params):
        ncomp = self.dim[0][0]
        self.params = self.params.from_natural_parameters(natural_params, ncomp)

    def sufficient_statistics_from_rvectors(self, rvecs):
        '''
        Real vector z = (x, y)
        \mu = x
        \sigma^2 = \exp(y)

        '''
        dim = self.dim[0] * self.dim[1]
        means = rvecs[:, :dim]
        log_precision = rvecs[:, dim:]
        precision = torch.exp(log_precision)
        return torch.cat([rvecs, rvecs**2], dim=-1)
