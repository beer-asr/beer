
'''Set of Normal densities with prior over the mean and
covariance matrix.

'''

import abc
from collections import namedtuple
import math
import torch

from .parameters import BayesianParameter
from .parameters import BayesianParameterSet
from .modelset import BayesianModelSet
from .normal import Normal
from .normal import NormalIsotropicCovariance
from .normal import NormalDiagonalCovariance
from .normal import NormalFullCovariance
from ..priors import IsotropicNormalGammaPrior
from ..priors import JointIsotropicNormalGammaPrior
from ..priors import NormalGammaPrior
from ..priors import JointNormalGammaPrior
from ..priors import NormalWishartPrior
from ..priors import JointNormalWishartPrior


NormalSetElement = namedtuple('NormalSetElement', ['mean', 'cov'])


class NormalSet(BayesianModelSet, metaclass=abc.ABCMeta):
    '''Set of Normal models.'''

    @staticmethod
    def create(mean, cov, size, prior_strength=1, noise_std=1.,
               cov_type='full', shared_cov=False):
        cov = torch.tensor(cov)
        if len(cov.shape) <= 1:
            std_dev = cov.sqrt()
        else:
            std_dev = cov.diag().sqrt()
        if shared_cov:
            return NormalSetSharedCovariance.create(mean, cov, size,
                                                    prior_strength,
                                                    noise_std * std_dev,
                                                    cov_type)
        else:
            return NormalSetNonSharedCovariance.create(mean, cov, size,
                                                       prior_strength,
                                                       noise_std * std_dev,
                                                       cov_type)

    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


########################################################################
# Normal set with no shared covariance matrix.
########################################################################

class NormalSetNonSharedCovariance(NormalSet, metaclass=abc.ABCMeta):

    @staticmethod
    def create(mean, cov, size, prior_strength=1, noise_std=1., cov_type='full'):
        normal = Normal.create(mean, cov, prior_strength, cov_type)
        prior = normal.mean_precision.prior
        posteriors = []
        dtype, device = mean.dtype, mean.device
        for i in range(size):
            noise = noise_std * torch.randn(len(mean), dtype=dtype, device=device)
            normal = Normal.create(mean + noise, cov, prior_strength, cov_type)
            posteriors.append(normal.mean_precision.posterior)

        # At this point, we are sure that the cov_type is valid.
        if cov_type == 'full':
            cls = NormalSetFullCovariance
        elif cov_type == 'diagonal':
            cls = NormalSetDiagonalCovariance
        else:
            cls = NormalSetIsotropicCovariance

        return cls(prior, posteriors)

    def __init__(self, prior, posteriors):
        super().__init__()
        self.means_precisions = BayesianParameterSet([
            BayesianParameter(prior, post)
            for post in posteriors
        ])

    def __len__(self):
        return len(self.means_precisions)

    @property
    def dim(self):
        return len(self.means_precisions[0].expected_value()[0])

    def mean_field_factorization(self):
        return [[*self.means_precisions]]

    def expected_log_likelihood(self, stats):
        nparams = self.means_precisions.expected_natural_parameters()
        return stats @ nparams.t() - .5 * self.dim * math.log(2 * math.pi)

    def marginal_log_likelihood(self, stats):
        m_llhs = []
        for model in self.means_precisions:
            post = mean_precision.posterior
            m_llhs.append(post.log_norm(post.natural_parameters + stats) \
                         - post.log_norm())
        return torch.cat(m_llhs, dim=-1)

    def accumulate(self, stats, weights):
        return dict(zip(self.means_precisions, torch.tensor(weights.t() @ stats)))


class NormalSetIsotropicCovariance(NormalSetNonSharedCovariance):
    '''Set of Normal models with isotropic covariance matrix.'''

    def __getitem__(self, key):
        mean, precision = self.means_precisions[key].expected_value()
        dtype, device = precision.dtype, precision.device
        cov = torch.eye(self.dim, dtype=dtype, device=device) / precision
        return NormalSetElement(mean=mean, cov=cov)

    @staticmethod
    def sufficient_statistics(data):
        return NormalIsotropicCovariance.sufficient_statistics(data)

    def marginal_log_likelihood(self, stats):
        m_llhs = []
        for param in self.means_precisions:
            cls = NormalIsotropicCovariance
            c_m_llhs = cls._marginal_log_likelihood(param.posterior, stats)
            m_llhs.append(c_m_llhs.view(-1, 1))
        return torch.cat(m_llhs, dim=-1)


class NormalSetDiagonalCovariance(NormalSetNonSharedCovariance):
    '''Set of Normal models with diagonal covariance matrix.'''

    def __getitem__(self, key):
        mean, precision = self.means_precisions[key].expected_value()
        cov =  (1. / precision).diag()
        return NormalSetElement(mean=mean, cov=cov)

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)

    def marginal_log_likelihood(self, stats):
        m_llhs = []
        for param in self.means_precisions:
            cls = NormalDiagonalCovariance
            c_m_llhs = cls._marginal_log_likelihood(param.posterior, stats)
            m_llhs.append(c_m_llhs.view(-1, 1))
        return torch.cat(m_llhs, dim=-1)


class NormalSetFullCovariance(NormalSetNonSharedCovariance):
    '''Set of Normal models with full covariance matrix.'''

    def __getitem__(self, key):
        mean, precision = self.means_precisions[key].expected_value()
        cov =  precision.inverse()
        return NormalSetElement(mean=mean, cov=cov)

    @staticmethod
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)

    def marginal_log_likelihood(self, stats):
        m_llhs = []
        for param in self.means_precisions:
            cls = NormalFullCovariance
            c_m_llhs = cls._marginal_log_likelihood(param.posterior, stats)
            m_llhs.append(c_m_llhs.view(-1, 1))
        return torch.cat(m_llhs, dim=-1)


########################################################################
# Normal set with shared covariance matrix.
########################################################################

class NormalSetSharedCovariance(NormalSet, metaclass=abc.ABCMeta):

    @staticmethod
    def create(mean, cov, size, prior_strength=1, noise_std=1., cov_type='full'):
        # Ensure the covariance is full.
        if len(cov.shape) == 1:
            if cov.shape[0] == 1:
                dtype, device = mean.dtype, mean.device
                full_cov = cov * torch.eye(len(mean), dtype=dtype, device=device)
            else:
                full_cov = cov.diag()
        else:
            full_cov = cov

        if cov_type == 'full':
            return NormalSetSharedFullCovariance.create(mean, full_cov, size,
                                                        prior_strength,
                                                        noise_std)
        elif cov_type == 'diagonal':
            return NormalSetSharedDiagonalCovariance.create(mean, full_cov, size,
                                                            prior_strength,
                                                            noise_std)
        elif cov_type == 'isotropic':
            return NormalSetSharedIsotropicCovariance.create(mean, full_cov, size,
                                                             prior_strength,
                                                             noise_std)
        else:
            raise ValueError('Unknown covariance type: "{cov_type}"'.format(
                cov_type=cov_type))

    def __init__(self, prior, posterior):
        super().__init__()
        self.means_precision = BayesianParameter(prior, posterior)

    def __len__(self):
        means, _ = self.means_precision.expected_value()
        return means.shape[0]

    @property
    def dim(self):
        means, _ = self.means_precision.expected_value()
        return means.shape[1]

    def mean_field_factorization(self):
        return [[self.means_precision]]

    def marginal_log_likelihood(self, stats):
        joint_nparams = self.means_precision.posterior.natural_parameters
        np1, np2 = self._split_natural_parameters(joint_nparams)
        np1 = torch.ones(len(np2), 1, dtype=np1.dtype,
                         device=np1.device) * np1.view(1, -1)
        nparams1 = torch.cat([
            np1[:, :-1],
            np2,
            np1[:, -1].view(-1, 1)
        ], dim=1)[None]

        new_stats = torch.cat([
            stats[:, :int(self.dim ** 2)],
            stats[:, int(self.dim ** 2):-1] / len(self),
            stats[:, -1].view(-1, 1)
        ], dim=-1)
        nparams2 = new_stats[:, None, :] + nparams1
        post = self.means_precision.posterior
        return post.joint_log_norm(nparams2) - post.joint_log_norm(nparams1)


class NormalSetSharedIsotropicCovariance(NormalSetSharedCovariance):
    '''Set of Normal density models with a shared isotropic covariance
    matrix.
    '''

    @classmethod
    def create(cls, mean, cov, size, prior_strength=1, noise_std=1.):
        dtype, device = mean.dtype, mean.device
        variance = cov.diag().max()
        scales = torch.ones(size, dtype=dtype, device=device)
        scales *= prior_strength
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate = torch.tensor(prior_strength * variance, dtype=dtype,
                            device=device)
        p_means = mean + torch.zeros(size, len(mean), dtype=dtype, device=device)
        means = mean +  noise_std * torch.randn(size, len(mean), dtype=dtype,
                                                device=device)
        prior = JointIsotropicNormalGammaPrior(p_means, scales, shape, rate)
        posterior = JointIsotropicNormalGammaPrior(means, scales, shape, rate)
        return cls(prior, posterior)

    def __getitem__(self, key):
        means, precision = self.means_precision.expected_value()
        dtype, device = precision.dtype, precision.device
        cov = torch.eye(self.dim, dtype=dtype, device=device) / precision
        return NormalSetElement(mean=means[key], cov=cov)

    def _split_natural_parameters(self, nparams):
        nparams1 = nparams[[0, -1]]
        nparams2 = torch.cat([
            nparams[1:self.dim * len(self) + 1].view(len(self), self.dim),
            nparams[-(len(self) + 1):-1].reshape(-1, 1)
        ], dim=-1)
        return nparams1, nparams2

    @staticmethod
    def sufficient_statistics(data):
        return NormalIsotropicCovariance.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        stats1, stats2 = stats[:, (0, -1)], stats[:, 1:-1]
        nparams = self.means_precision.expected_natural_parameters()
        nparams1, nparams2 = self._split_natural_parameters(nparams)
        exp_llhs = (stats1 @ nparams1)[:, None] + stats2 @ nparams2.t()
        exp_llhs -= .5 * self.dim * math.log(2 * math.pi)
        return exp_llhs

    def marginal_log_likelihood(self, stats):
        m_llhs = []
        for mean_precision in self.means_precisions:
            post = mean_precision.posterior
            m_llhs.append(post.log_norm(post.natural_parameters + stats) \
                         - post.log_norm())
        return torch.cat(m_llhs, dim=-1)

    def accumulate(self, stats, resps):
        dtype, device = stats.dtype, stats.device
        w_stats = resps.t() @ stats
        acc_stats = torch.cat([
            w_stats[:, 0].sum().view(1),
            w_stats[:, 1: 1 + self.dim].contiguous().view(-1),
            w_stats[:, -2].view(-1),
            w_stats[:, -1].sum().view(1)
        ], dim=0)
        return {self.means_precision: torch.tensor(acc_stats)}


class NormalSetSharedDiagonalCovariance(NormalSetSharedCovariance):
    '''Set of Normal density models with a shared full covariance
    matrix.
    '''

    @classmethod
    def create(cls, mean, cov, size, prior_strength=1, noise_std=1.):
        dtype, device = mean.dtype, mean.device
        variance = cov.diag()
        scales = torch.ones(size, dtype=dtype, device=device)
        scales *= prior_strength
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rates = variance * prior_strength
        p_means = mean + torch.zeros(size, len(mean), dtype=dtype, device=device)
        means = mean +  noise_std * torch.randn(size, len(mean), dtype=dtype,
                                                device=device)
        prior = JointNormalGammaPrior(p_means, scales, shape, rates)
        posterior = JointNormalGammaPrior(means, scales, shape, rates)
        return cls(prior, posterior)

    def __getitem__(self, key):
        means, precision = self.means_precision.expected_value()
        cov = (1 / precision).diag()
        return NormalSetElement(mean=means[key], cov=cov)

    def _split_stats(self, stats):
        stats1 = torch.cat([stats[:,:self.dim], stats[:,-1].view(-1, 1)], dim=-1)
        stats2 = stats[:, self.dim:-1]
        return stats1, stats2

    def _split_natural_parameters(self, nparams):
        nparams1 = torch.cat([nparams[:self.dim], nparams[-1].view(1)], dim=-1)
        nparams2 = torch.cat([
            nparams[self.dim:self.dim * len(self) + self.dim].view(len(self), self.dim),
            nparams[-(len(self) + 1):-1].reshape(-1, 1)
        ], dim=-1)
        return nparams1, nparams2

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        stats1, stats2 = self._split_stats(stats)
        nparams = self.means_precision.expected_natural_parameters()
        nparams1, nparams2 = self._split_natural_parameters(nparams)
        exp_llhs = (stats1 @ nparams1)[:, None] + stats2 @ nparams2.t()
        exp_llhs -= .5 * self.dim * math.log(2 * math.pi)
        return exp_llhs

    def accumulate(self, stats, resps):
        w_stats = resps.t() @ stats
        acc_stats = torch.cat([
            w_stats[:, :self.dim].sum(dim=0),
            w_stats[:, self.dim: 2 * self.dim].contiguous().view(-1),
            w_stats[:, -2].view(-1),
            w_stats[:, -1].sum().view(1)
        ], dim=0)
        return {self.means_precision: torch.tensor(acc_stats)}


class NormalSetSharedFullCovariance(NormalSetSharedCovariance):
    '''Set of Normal density models with a  shared covariance matrix.'''

    @classmethod
    def create(cls, mean, cov, size, prior_strength=1, noise_std=1.):
        dtype, device = mean.dtype, mean.device
        scales = torch.ones(size, dtype=dtype, device=device)
        scales *= prior_strength
        dof = torch.tensor(prior_strength + len(mean) - 1, dtype=dtype,
                           device=device)
        scale_matrix = torch.inverse(cov *  dof)
        p_means = mean + torch.zeros(size, len(mean), dtype=mean.dtype,
                                    device=mean.device)
        p_means = mean + torch.zeros(size, len(mean), dtype=dtype, device=device)
        means = mean +  noise_std * torch.randn(size, len(mean), dtype=dtype,
                                                device=device)
        prior = JointNormalWishartPrior(p_means, scales, scale_matrix, dof)
        posterior = JointNormalWishartPrior(means, scales, scale_matrix, dof)
        return cls(prior, posterior)

    def __getitem__(self, key):
        means, precision = self.means_precision.expected_value()
        cov = precision.inverse()
        return NormalSetElement(mean=means[key], cov=cov)

    def _split_stats(self, stats):
        stats1 = torch.cat([stats[:,:self.dim ** 2], stats[:,-1].view(-1, 1)], dim=-1)
        stats2 = stats[:, self.dim**2:-1]
        return stats1, stats2

    def _split_natural_parameters(self, nparams):
        nparams1 = torch.cat([nparams[:self.dim**2], nparams[-1].view(1)], dim=-1)
        start = self.dim**2
        end = start + len(self) * self.dim
        nparams2 = torch.cat([
            nparams[start:end].view(len(self), self.dim),
            nparams[-(len(self) + 1):-1].reshape(-1, 1)
        ], dim=-1)
        return nparams1, nparams2

    @staticmethod
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        stats1, stats2 = self._split_stats(stats)
        nparams = self.means_precision.expected_natural_parameters()
        nparams1, nparams2 = self._split_natural_parameters(nparams)
        exp_llhs = (stats1 @ nparams1)[:, None] + stats2 @ nparams2.t()
        exp_llhs -= .5 * self.dim * math.log(2 * math.pi)
        return exp_llhs

    def accumulate(self, stats, resps):
        w_stats = resps.t() @ stats
        acc_stats = torch.cat([
            w_stats[:, :self.dim**2].sum(dim=0),
            w_stats[:, self.dim**2: self.dim + (self.dim**2)].contiguous().view(-1),
            w_stats[:, -2].view(-1),
            w_stats[:, -1].sum().view(1)
        ], dim=0)
        return {self.means_precision: torch.tensor(acc_stats)}


__all__ = ['NormalSet']
