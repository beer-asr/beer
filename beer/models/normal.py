
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

    @property
    @abc.abstractmethod
    def mean(self):
        pass

    @property
    @abc.abstractmethod
    def cov(self):
        pass

    def float(self):
        return self.__class__(
            self.mean_prec_param.prior.float(),
            self.mean_prec_param.posterior.float()
        )

    def double(self):
        return self.__class__(
            self.mean_prec_param.prior.double(),
            self.mean_prec_param.posterior.double()
        )

    def to(self, device):
        return self.__class__(
            self.mean_prec_param.prior.to(device),
            self.mean_prec_param.posterior.to(device)
        )

class NormalIsotropicCovariance(Normal):
    '''Normal model with isotropic covariance matrix.'''

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @property
    def mean(self):
        np1, np2, _, _ = \
            self.mean_prec_param.expected_value(concatenated=False)
        return np2 / (-2 * np1)

    @property
    def cov(self):
        np1, np2, _, _ = \
            self.mean_prec_param.expected_value(concatenated=False)
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

    def forward(self, s_stats):
        feadim = s_stats.size(1) - 3
        exp_llh = s_stats @ self.mean_prec_param.expected_value()
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        dtype, device = mean.dtype, mean.device
        return torch.cat([
            ((mean ** 2).sum(dim=1) + var.sum(dim=1)).view(-1, 1),
            mean,
            torch.ones(len(mean), 2, dtype=dtype, device=device)
        ], dim=-1)

    def expected_natural_params(self, mean, var, nsamples=1):
        dtype, device = mean.dtype, mean.device
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)
        np1, np2, np3, np4 = \
            self.mean_prec_param.expected_value(concatenated=False)
        feadim = len(np2)
        exp_nparams = torch.cat([
            np1 * torch.ones(feadim,  dtype=dtype, device=device),
            np2,
            np3 * torch.ones(feadim,  dtype=dtype, device=device) / feadim,
            np4 * torch.ones(feadim,  dtype=dtype, device=device) / feadim,
        ])
        ones = torch.ones(s_stats.size(0), exp_nparams.size(0), dtype=dtype,
                          device=device)
        return ones * exp_nparams, s_stats


class NormalDiagonalCovariance(Normal):
    '''Normal model with diagonal covariance matrix.'''

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @property
    def mean(self):
        np1, np2, _, _ = \
            self.mean_prec_param.expected_value(concatenated=False)
        return np2 / (-2 * np1)

    @property
    def cov(self):
        np1, _, _, _ = \
            self.mean_prec_param.expected_value(concatenated=False)
        return torch.diag(1/(-2 * np1))

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([data ** 2, data, torch.ones_like(data),
                          torch.ones_like(data)], dim=-1)

    def forward(self, s_stats):
        feadim = .25 * s_stats.size(1)
        exp_llh = s_stats @ self.mean_prec_param.expected_value()
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([(mean ** 2) + var, mean, torch.ones_like(mean),
                          torch.ones_like(mean)], dim=-1)

    def expected_natural_params(self, mean, var, nsamples=1):
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)
        nparams = self.mean_prec_param.expected_value()
        ones = torch.ones(s_stats.size(0), nparams.size(0), dtype=s_stats.dtype,
                          device=s_stats.device)
        return ones * nparams, s_stats


class NormalFullCovariance(Normal):
    '''Normal model with full covariance matrix.'''

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @property
    def mean(self):
        np1, np2, _, _ = \
            self.mean_prec_param.expected_value(concatenated=False)
        return torch.inverse(-2 * np1) @ np2

    @property
    def cov(self):
        np1, _, _, _ = \
            self.mean_prec_param.posterior.split_sufficient_statistics(
                self.mean_prec_param.expected_value()
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

    def forward(self, s_stats):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - s_stats.size(1))))
        exp_llh = s_stats @ self.mean_prec_param.expected_value()
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}


def create(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    covariance_type = model_conf['covariance']
    noise_std = model_conf['noise_std']
    prior_strength = model_conf['prior_strength']
    rand_mean = mean +  noise_std * torch.randn(len(mean), dtype=dtype,
                                                device=device)
    if covariance_type == 'isotropic':
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate =  torch.tensor(prior_strength * variance.sum(),
                             dtype=dtype, device=device)
        prior = IsotropicNormalGammaPrior(mean, scale, shape, rate)
        posterior = IsotropicNormalGammaPrior(rand_mean, scale, shape, rate)
        return NormalIsotropicCovariance(prior, posterior)
    elif covariance_type == 'diagonal':
        scale = torch.ones_like(mean) * prior_strength
        shape = torch.ones_like(mean) * prior_strength
        rate = prior_strength * variance
        prior = NormalGammaPrior(mean, scale, shape, rate)
        posterior = NormalGammaPrior(rand_mean, scale, shape, rate)
        return NormalDiagonalCovariance(prior, posterior)
    elif covariance_type == 'full':
        cov = torch.diag(variance)
        scale = prior_strength
        dof = prior_strength + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posterior = NormalWishartPrior(rand_mean, scale, scale_matrix, dof)
        return NormalFullCovariance(prior, posterior)
    else:
        raise ValueError('Unknown covariance type: {}'.format(covariance_type))


__all__ = [
    'NormalIsotropicCovariance',
    'NormalDiagonalCovariance',
    'NormalFullCovariance'
]
