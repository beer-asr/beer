import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily


__all__ = ['NormalDiagonalCovariance', 'NormalDiagonalCovarianceStdParams']


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

    @property
    def conjugate_sufficient_statistics_dim(self):
        return self.dim

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
            self.params.diag_cov
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

    def sufficient_statistics_from_rvectors(self, rvecs):
        '''
        Real vector z = (x, y)
        \mu = x

        '''
        return torch.cat([rvecs, rvecs**2], dim=-1)
