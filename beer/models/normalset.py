
'''Set of Normal densities with prior over the mean and
covariance matrix.

'''

import abc
from collections import namedtuple
import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .bayesmodel import BayesianModelSet
from .normal import Normal
from .normal import NormalIsotropicCovariance
from .normal import NormalDiagonalCovariance
from .normal import NormalFullCovariance
from ..priors import IsotropicNormalGammaPrior
from ..priors import JointIsotropicNormalGammaPrior
from ..priors import NormalGammaPrior
from ..priors import JointNormalGammaPrior
from ..priors import NormalWishartPrior
from ..expfamilyprior import JointNormalWishartPrior


NormalSetElement = namedtuple('NormalSetElement', ['mean', 'cov'])


def _create_prior_posterior(mean, cov, size, prior_strength, noise_std, cov_type):
    normal = Normal.create(mean, cov, prior_strength, cov_type)
    prior = normal.mean_precision.prior
    posteriors = []
    dtype, device = mean.dtype, mean.device
    for i in range(size):
        noise = noise_std * torch.randn(len(mean), dtype=dtype, device=device)
        normal = Normal.create(mean + noise, cov, prior_strength, cov_type)
        posteriors.append(normal.mean_precision.posterior)

    return prior, posteriors


class NormalSet(BayesianModelSet, metaclass=abc.ABCMeta):
    '''Set of Normal models.'''

    @staticmethod
    def create(mean, cov, size, prior_strength=1, noise_std=1.,
               cov_type='full', shared_cov=False):
        if shared_cov:
            return NormalSetSharedCovariance.create(mean, cov, size,
                                                    prior_strength,
                                                    noise_std, cov_type)
        else:
            return NormalSetNonSharedCovariance.create(mean, cov, size,
                                                       prior_strength,
                                                       noise_std,cov_type)

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

    def accumulate(self, stats, weights):
        return dict(zip(self.means_precisions, weights.t() @ stats))


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


class NormalSetDiagonalCovariance(NormalSetNonSharedCovariance):
    '''Set of Normal models with diagonal covariance matrix.'''

    def __getitem__(self, key):
        mean, precision = self.means_precisions[key].expected_value()
        cov =  (1. / precision).diag()
        return NormalSetElement(mean=mean, cov=cov)

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)


class NormalSetFullCovariance(NormalSetNonSharedCovariance):
    '''Set of Normal models with full covariance matrix.'''

    def __getitem__(self, key):
        mean, precision = self.means_precisions[key].expected_value()
        cov =  precision.inverse()
        return NormalSetElement(mean=mean, cov=cov)

    @staticmethod
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)


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

    def accumulate(self, stats, resps):
        dtype, device = stats.dtype, stats.device
        w_stats = resps.t() @ stats
        acc_stats = torch.cat([
            w_stats[:, 0].sum().view(1),
            w_stats[:, 1: 1 + self.dim].contiguous().view(-1),
            w_stats[:, -2].view(-1),
            w_stats[:, -1].sum().view(1)
        ], dim=0)
        return {self.means_precision: acc_stats}


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
        nparams1 = nparams[[0, -1]]
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
        dtype, device = stats.dtype, stats.device
        w_stats = resps.t() @ stats
        acc_stats = torch.cat([
            w_stats[:, :self.dim].sum(dim=0),
            w_stats[:, self.dim: 2 * self.dim].contiguous().view(-1),
            w_stats[:, -2].view(-1),
            w_stats[:, -1].sum().view(1)
        ], dim=0)
        return {self.means_precision: acc_stats}

    def forward(self, s_stats):
        s_stats1, s_stats2 = s_stats
        feadim = s_stats1.size(1) // 2
        params = self._expected_nparams()
        retval = (s_stats1 @ params[0])[:, None] + s_stats2 @ params[1].t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval


class NormalSetSharedFullCovariance(NormalSetSharedCovariance):
    '''Set of Normal density models with a  shared covariance matrix.'''

    @classmethod
    def create(cls, mean, variance, size, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        cov = torch.diag(variance)
        scales = torch.ones(size, dtype=dtype, device=device)
        scales *= prior_strength
        dof = prior_strength + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        p_means = mean + torch.zeros(size, len(mean), dtype=mean.dtype,
                                    device=mean.device)
        means = mean + noise_std * torch.randn(size, len(mean), dtype=dtype,
                                                device=device)
        prior = JointNormalWishartPrior(p_means, scales, scale_matrix, dof)
        posteriors = JointNormalWishartPrior(means, scales, scale_matrix, dof)
        return cls(prior, posteriors)

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_precision = BayesianParameter(prior, posterior)

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_precision.expected_value(concatenated=False)
        cov = torch.inverse(-2 * np1)
        mean = cov @ np2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return self.means_precision.posterior.ncomp

    def reset_class_means(self, n_classes, noise_std=0.1, prior_strength=1.):
        '''Create a new set of class means randomly initialized.

        Args:
            n_classes (int): number of components.
            noise_std (float): Standard deviation of the noise for the
                random initialization.
            prior_strength (float): Strength of the class means' prior.

        '''
        dtype, device = self[0].mean.dtype, self[0].mean.device

        # Expected value of the mean/covariance matrix w.r.t. the prior
        # distribution.
        np1, np2, _, _ = \
            self.means_precision.prior.split_sufficient_statistics(
                self.means_precision.prior.expected_sufficient_statistics)
        prior_cov = torch.inverse(-2 * np1)
        prior_mean = prior_cov @ np2[0]
        print(prior_mean)

        # Dimension of the data.
        dim = len(prior_mean)

        # Expected value of the covariance matrix w.r.t. the posterior
        # distribution.
        cov = self[0].cov

        # Degree of freedom of the posterior Wishart distribution.
        dof = self.means_precision.posterior.natural_hparams[-1] + dim

        # Generate new parameters for the prior/posterior distribution.
        scales = torch.ones(n_classes, dtype=dtype, device=device)
        scales *= prior_strength
        scale_matrix = torch.inverse(cov *  dof)
        p_means = prior_mean + torch.zeros(n_classes, dim, dtype=dtype,
                                           device=device)
        U = torch.potrf(cov)
        means = prior_mean + noise_std *  torch.randn(n_classes, dim, dtype=dtype,
                                                     device=device) @ U
        prior = JointNormalWishartPrior(p_means, scales, scale_matrix, dof)
        posterior = JointNormalWishartPrior(means, scales, scale_matrix, dof)
        self.means_precision = BayesianParameter(prior, posterior)

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_precision.expected_value(concatenated=False)
        return np1.view(-1), \
            torch.cat([np2, np3[:, None]], dim=1), np4

    def mean_field_factorization(self):
        return [[self.means_precision]]

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = (data[:, :, None] * data[:, None, :]).view(len(data), -1)
        s_stats2 = torch.cat([data, torch.ones(data.size(0), 1,
                                               dtype=data.dtype,
                                               device=data.device)], dim=1)
        return s_stats1, s_stats2

    def forward(self, s_stats):
        s_stats1, s_stats2 = s_stats
        feadim = int(math.sqrt(s_stats1.size(1)))
        params = self._expected_nparams()
        retval = (s_stats1 @ params[0])[:, None] + \
            s_stats2 @ params[1].t() + params[2]
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        s_stats1, s_stats2, weights = *s_stats, parent_msg
        feadim = int(math.sqrt(s_stats1.size(1)))
        acc_stats = torch.cat([
            s_stats1.sum(dim=0),
            (weights.t() @ s_stats2[:, :feadim]).view(-1),
            weights.sum(dim=0),
            len(weights) * torch.ones(1, dtype=weights.dtype, device=weights.device)
        ])
        return {self.means_precision: acc_stats}


def create(model_conf, mean, variance, create_model_handle):
    size = model_conf['size']
    covariance_type = model_conf['covariance']
    shared_cov = model_conf['shared_covariance']
    noise_std = model_conf['noise_std']
    prior_strength = model_conf['prior_strength']
    if covariance_type == 'isotropic':
        if shared_cov:
            return NormalSetSharedIsotropicCovariance.create(
                mean, variance, size, prior_strength, noise_std)
        else:
            return NormalSetIsotropicCovariance.create(
                mean, variance, size, prior_strength, noise_std)
    elif covariance_type == 'diagonal':
        if shared_cov:
            return NormalSetSharedDiagonalCovariance.create(
                mean, variance, size, prior_strength, noise_std)
        else:
            return NormalSetDiagonalCovariance.create(
                mean, variance, size, prior_strength, noise_std)
    elif covariance_type == 'full':
        if shared_cov:
            return NormalSetSharedFullCovariance.create(
                mean, variance, size, prior_strength, noise_std)
        else:
            return NormalSetFullCovariance.create(
                mean, variance, size, prior_strength, noise_std)
    else:
        raise ValueError('Unknown covariance type: {}'.format(covariance_type))


__all__ = ['NormalSet']
