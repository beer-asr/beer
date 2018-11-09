
'''Bayesian Normal distribution with prior over the mean and
covariance matrix.
'''

import abc
import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel
from ..priors import IsotropicNormalGammaPrior
from ..priors import NormalGammaPrior
from ..priors import NormalWishartPrior
from ..priors.normalwishart import _logdet


class Normal(BayesianModel):
    '''Normal model with prior over the mean and variance parameter.

    Attributes:
        mean: Mean parameter.
        cov: Covariance parameter.

    '''

    @staticmethod
    def create(mean, cov, prior_strength=1., cov_type='full'):
        '''Create a Normal model.

        Args:
            mean (``torch.Tensor[dim]``): Initial mean of the model.
            cov (``torch.Tensor[dim, dim]`` or ``torch.Tensor[dim]`` or
                scalar):
                Initial covariance matrix. Can be specified as a
                dense/diagonal matrix or a scalar.
            cov_type (str): Type of the covariance matrix. Can be
                "full", "diagonal", or "isotropic".

        Returns:
            :any:`Normal`

        '''
        # Ensure the covariance is full.
        if len(cov.shape) == 1:
            if cov.shape[0] == 1:
                dtype, device = mean.dtype, mean.device
                full_cov = cov * torch.eye(len(mean), dtype=dtype, device=device)
            else:
                full_cov = cov.diag()
        else:
            full_cov = cov

        try:
            normal = cov_types[cov_type](mean, full_cov, prior_strength)
        except KeyError:
            raise ValueError('Unknown covariance type: "{cov_type}"'.format(
                cov_type=cov_type))
        return normal

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_precision = BayesianParameter(prior, posterior)

    @property
    def dim(self):
        mean, _ = self.mean_precision.expected_value()
        return len(mean)

    @property
    def mean(self):
        return self.mean_precision.expected_value()[0]

    @property
    @abc.abstractmethod
    def cov(self):
        pass

    def mean_field_factorization(self):
        return [[self.mean_precision]]

    def expected_log_likelihood(self, stats):
        nparams = self.mean_precision.expected_natural_parameters()
        return (stats * nparams[None]).sum(dim=-1)  -.5 * self.dim * math.log(2 * math.pi)

    def marginal_log_likelihood(prior, stats):
        return self._marginal_log_likelihood(self.mean_precision.posterior, stats)

    def accumulate(self, stats, parent_msg=None):
        return {self.mean_precision: torch.tensor(stats.sum(dim=0))}


class NormalIsotropicCovariance(Normal):
    '''Normal model with isotropic covariance matrix.'''

    @classmethod
    def create(cls, mean, cov, prior_strength=1.):
        variance = cov.diag().max()
        dtype, device = mean.dtype, mean.device
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate =  torch.tensor(prior_strength * variance, dtype=dtype, device=device)
        prior = IsotropicNormalGammaPrior(mean, scale, shape, rate)
        posterior = IsotropicNormalGammaPrior(mean, scale, shape, rate)
        return cls(prior, posterior)

    @property
    def cov(self):
        _, precision = self.mean_precision.expected_value()
        dtype, device = precision.dtype, precision.device
        return torch.eye(self.dim, dtype=dtype, device=device) / precision

    @staticmethod
    def sufficient_statistics(data):
        dim, dtype, device = data.shape[1], data.dtype, data.device
        return torch.cat([
            -.5 * torch.sum(data**2, dim=-1).reshape(-1, 1),
            data,
            -.5 * torch.ones(len(data), 1, dtype=dtype, device=device),
            .5 * dim * torch.ones(len(data), 1, dtype=dtype, device=device),
        ], dim=-1)


    @staticmethod
    def _marginal_log_likelihood(prior, stats):
        mean, scale, shape, rate = prior.to_std_parameters()
        mean, scale, shape, rate = mean.view(-1), scale.view(-1), \
                                    shape.view(-1), rate.view(-1)

        dim = len(mean)
        alpha = 1 + 1 / scale
        lnorm = (torch.lgamma(shape + .5 * dim) - torch.lgamma(shape))
        lnorm -= .5 * dim * (alpha.log() + math.log(2 * math.pi))
        lnorm -= .5 * dim * rate.log()
        data = stats[:, 1: 1 + dim]
        kernel = 1 + (data - mean).pow(2).sum(dim=-1) / (2 * alpha * rate)
        #import pdb; pdb.set_trace()
        return - (shape + .5 * dim) * kernel.log() + lnorm

        dim = len(mean)
        alpha = 1 + 1 / scale
        lnorm = dim * (torch.lgamma(shape + .5 ) - torch.lgamma(shape))
        lnorm -= .5 * dim * (alpha.log() + math.log(2 * math.pi))
        lnorm -= .5 * dim * rate.log().sum()
        data = stats[:, 1: 1 + dim]
        kernels = 1 + (data - mean).pow(2) / (2 * alpha * rate)
        return - (shape + .5) * kernels.log().sum(dim=-1) + lnorm


class NormalDiagonalCovariance(Normal):
    '''Normal model with diagonal covariance matrix.'''

    @classmethod
    def create(cls, mean, cov, prior_strength=1.):
        variance = cov.diag()
        dtype, device = mean.dtype, mean.device
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rates = prior_strength * variance
        prior = NormalGammaPrior(mean, scale, shape, rates)
        posterior = NormalGammaPrior(mean, scale, shape, rates)
        return cls(prior, posterior)

    @property
    def cov(

    ):
        _, precision = self.mean_precision.expected_value()
        return (1. / precision).diag()

    @staticmethod
    def sufficient_statistics(data):
        dtype, device = data.dtype, data.device
        return torch.cat([
            -.5 * data.pow(2),
            data,
            -.5 * torch.ones(len(data), 1, dtype=dtype, device=device),
            .5 * torch.ones(len(data), 1, dtype=dtype, device=device),
        ], dim=-1)

    @staticmethod
    def _marginal_log_likelihood(prior, stats):
        mean, scale, shape, rates = prior.to_std_parameters()
        mean, scale, shape, rates = mean.view(-1), scale.view(-1), \
                                    shape.view(-1), rates.view(-1)
        dim = len(mean)
        alpha = 1 + 1 / scale
        lnorm = dim * (torch.lgamma(shape + .5 ) - torch.lgamma(shape))
        lnorm -= .5 * dim * (alpha.log() + math.log(2 * math.pi))
        lnorm -= .5 * rates.log().sum()
        data = stats[:, dim: 2 * dim]
        kernels = 1 + (data - mean).pow(2) / (2 * alpha * rates)
        return - (shape + .5) * kernels.log().sum(dim=-1) + lnorm


class NormalFullCovariance(Normal):
    '''Normal model with full covariance matrix.'''

    @classmethod
    def create(cls, mean, cov, prior_strength=1.):
        dtype, device = mean.dtype, mean.device
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        dof = torch.tensor(prior_strength + len(mean) - 1, dtype=dtype, device=device)
        scale_matrix = cov.inverse() / dof
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posterior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        return cls(prior, posterior)

    @property
    def cov(self):
        _, precision = self.mean_precision.expected_value()
        return precision.inverse()

    @staticmethod
    def sufficient_statistics(data):
        dtype, device = data.dtype, data.device
        return torch.cat([
            (-.5 * data[:, :, None] * data[:, None, :]).view(len(data), -1),
            data,
            -.5 * torch.ones(data.size(0), 1, dtype=dtype, device=device),
            .5 * torch.ones(data.size(0), 1, dtype=dtype, device=device),
        ], dim=-1)

    @staticmethod
    def _marginal_log_likelihood(prior, stats):
        mean, k, W, dof = prior.to_std_parameters()
        dim = mean.shape[-1]
        mean, k, W, dof = mean.view(-1), k.view(-1), W.view(dim, dim), dof.view(-1)
        alpha = 1 + 1/k

        lnorm = .5 * _logdet(W)
        lnorm += torch.lgamma(.5 * (dof + 1))
        lnorm -= torch.lgamma(.5 * (dof - dim + 1))
        lnorm -= .5 * dim * torch.log(alpha * math.pi)

        quad_mean = (torch.ger(mean, mean).view(-1) * W.view(-1)).sum()
        vec_params = torch.cat([
            2 * W.view(-1),
            W @ mean,
            2 * quad_mean.view(1)
        ])
        kernel = 1 + stats[:, :-1] @ (-vec_params) / alpha

        return -.5 * (dof + 1) * kernel.log() + lnorm


cov_types = {
    'full': NormalFullCovariance.create,
    'diagonal': NormalDiagonalCovariance.create,
    'isotropic': NormalIsotropicCovariance.create
}


__all__ = [
    'Normal',
    'NormalIsotropicCovariance',
    'NormalDiagonalCovariance',
    'NormalFullCovariance'
]
