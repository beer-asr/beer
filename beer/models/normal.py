
'''Bayesian Normal distribution with prior over the mean and
covariance matrix.
'''

import abc
import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel
from ..expfamilyprior import IsotropicNormalGammaPrior
from ..expfamilyprior import NormalGammaPrior
from ..expfamilyprior import NormalWishartPrior


class Normal(BayesianModel):
    '''Normal model with prior over the mean and variance parameter.

    Attributes:
        mean: Mean parameter.
        cov: Covariance parameter.

    '''
    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_precision = BayesianParameter(prior, posterior)
        self._feadim = len(self.mean)

    @property
    def mean(self):
        return self._get_mean(self.mean_precision)

    @property
    def cov(self):
        return self._get_cov(self.mean_precision)

    def mean_field_factorization(self):
        return [[self.mean_precision]]

    def forward(self, s_stats):
        exp_llh = s_stats @ self.mean_precision.expected_value()
        exp_llh -= .5 * self._feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_precision: s_stats.sum(dim=0)}

    @staticmethod
    @abc.abstractmethod
    def _get_mean(param):
        pass

    @staticmethod
    @abc.abstractmethod
    def _get_cov(param):
        pass


class NormalIsotropicCovariance(Normal):
    '''Normal model with isotropic covariance matrix.'''

    @classmethod
    def create(cls, mean, variance, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        rand_mean = mean +  noise_std * torch.randn(len(mean), dtype=dtype,
                                                device=device)
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate =  torch.tensor(prior_strength * variance.mean(),
                             dtype=dtype, device=device)
        prior = IsotropicNormalGammaPrior(mean, scale, shape, rate)
        posterior = IsotropicNormalGammaPrior(rand_mean, scale, shape, rate)
        return cls(prior, posterior)

    @staticmethod
    def _get_mean(param):
        np1, np2, _, _ = param.expected_value(concatenated=False)
        return np2 / (-2 * np1)

    @staticmethod
    def _get_cov(param):
        np1, np2, _, _ = param.expected_value(concatenated=False)
        dtype, device = np1.dtype, np1.device
        return torch.eye(len(np2), dtype=dtype, device=device) / (-2 * np1)

    @staticmethod
    def sufficient_statistics(data):
        dtype, device = data.dtype, data.device
        return torch.cat([
            (data ** 2).sum(dim=1).view(-1, 1),
            data,
            torch.ones(len(data), 2, dtype=dtype, device=device)
        ], dim=-1)


class NormalDiagonalCovariance(Normal):
    '''Normal model with diagonal covariance matrix.'''

    @classmethod
    def create(cls, mean, variance, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        rand_mean = mean +  noise_std * torch.randn(len(mean), dtype=dtype,
                                                device=device)
        scale = torch.ones_like(mean) * prior_strength
        shape = torch.ones_like(mean) * prior_strength
        rate = prior_strength * variance
        prior = NormalGammaPrior(mean, scale, shape, rate)
        posterior = NormalGammaPrior(rand_mean, scale, shape, rate)
        return cls(prior, posterior)

    @staticmethod
    def _get_mean(param):
        np1, np2, _, _ = param.expected_value(concatenated=False)
        return np2 / (-2 * np1)

    @staticmethod
    def _get_cov(param):
        np1, _, _, _ = param.expected_value(concatenated=False)
        return torch.diag(1/(-2 * np1))

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([data ** 2, data, torch.ones_like(data),
                          torch.ones_like(data)], dim=-1)


class NormalFullCovariance(Normal):
    '''Normal model with full covariance matrix.'''

    @classmethod
    def create(cls, mean, variance, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        rand_mean = mean +  noise_std * torch.randn(len(mean), dtype=dtype,
                                                device=device)
        cov = torch.diag(variance)
        scale = prior_strength
        dof = prior_strength + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posterior = NormalWishartPrior(rand_mean, scale, scale_matrix, dof)
        return cls(prior, posterior)

    @staticmethod
    def _get_mean(param):
        np1, np2, _, _ = param.expected_value(concatenated=False)
        return torch.inverse(-2 * np1) @ np2

    @staticmethod
    def _get_cov(param):
        np1, _, _, _ = param.posterior.split_sufficient_statistics(
            param.expected_value()
        )
        return torch.inverse(-2 * np1)

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([
            (data[:, :, None] * data[:, None, :]).view(len(data), -1),
            data, torch.ones(data.size(0), 1, dtype=data.dtype,
                             device=data.device),
            torch.ones(data.size(0), 1, dtype=data.dtype, device=data.device)
        ], dim=-1)


def create(model_conf, mean, variance, create_model_handle):
    covariance_type = model_conf['covariance']
    noise_std = model_conf['noise_std']
    prior_strength = model_conf['prior_strength']
    if covariance_type == 'isotropic':
        return NormalIsotropicCovariance.create(mean, variance, prior_strength,
                                                noise_std)
    elif covariance_type == 'diagonal':
        return NormalDiagonalCovariance.create(mean, variance, prior_strength,
                                               noise_std)
    elif covariance_type == 'full':
        return NormalFullCovariance.create(mean, variance, prior_strength,
                                           noise_std)
    else:
        raise ValueError('Unsupported covariance type: {}'.format(
            covariance_type))


__all__ = [
    'NormalIsotropicCovariance',
    'NormalDiagonalCovariance',
    'NormalFullCovariance'
]
