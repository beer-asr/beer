
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
from .normal import NormalIsotropicCovariance
from .normal import NormalDiagonalCovariance
from .normal import NormalFullCovariance
from ..expfamilyprior import JointIsotropicNormalGammaPrior
from ..expfamilyprior import IsotropicNormalGammaPrior
from ..expfamilyprior import NormalGammaPrior
from ..expfamilyprior import JointNormalGammaPrior
from ..expfamilyprior import NormalWishartPrior
from ..expfamilyprior import JointNormalWishartPrior


NormalSetElement = namedtuple('NormalSetElement', ['mean', 'cov'])


class NormalSet(BayesianModelSet):
    '''Set of Normal models.'''

    def __init__(self, prior, posteriors):
        super().__init__()
        self.normals = BayesianParameterSet([
            BayesianParameter(prior, post)
            for post in posteriors
        ])
        self._feadim = len(self[0].mean)

    def __getitem__(self, key):
        param = self.normals[key]
        mean = self._get_mean(param)
        cov =  self._get_cov(param)
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return len(self.normals)

    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value()[None]
                          for param in self.normals], dim=0)

    def mean_field_factorization(self):
        return [[*self.normals]]

    def forward(self, s_stats):
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * self._feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        return dict(zip(self.normals, weights.t() @ s_stats))

    @staticmethod
    @abc.abstractmethod
    def _get_mean(param):
        pass
    @staticmethod
    @abc.abstractmethod
    def _get_cov(param):
        pass

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(data):
        pass


class NormalSetIsotropicCovariance(NormalSet):
    '''Set of Normal models with isotropic covariance matrix.'''

    @classmethod
    def create(cls, mean, variance, size, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate = torch.tensor(prior_strength * variance.sum(), dtype=dtype,
                            device=device)
        prior = IsotropicNormalGammaPrior(mean, scale, shape, rate)
        rand_means = noise_std * torch.randn(size, len(mean), dtype=dtype,
                                             device=device) + mean
        posteriors = [IsotropicNormalGammaPrior(rand_means[i], scale, shape,
                                                rate)
                        for i in range(size)]
        return cls(prior, posteriors)

    @staticmethod
    def _get_mean(param):
        return NormalIsotropicCovariance._get_mean(param)

    @staticmethod
    def _get_cov(param):
        return NormalIsotropicCovariance._get_cov(param)

    @staticmethod
    def sufficient_statistics(data):
        return NormalIsotropicCovariance.sufficient_statistics(data)


class NormalSetDiagonalCovariance(NormalSet):
    '''Set of Normal models with diagonal covariance matrix.'''

    @classmethod
    def create(cls, mean, variance, size, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        scale = torch.ones_like(mean) * prior_strength
        shape = torch.ones_like(mean) * prior_strength
        rate = prior_strength * variance
        prior = NormalGammaPrior(mean, scale, shape, rate)
        rand_means = noise_std * torch.randn(size, len(mean), dtype=dtype,
                                                device=device) + mean
        posteriors = [NormalGammaPrior(rand_means[i], scale, shape, rate)
                        for i in range(size)]
        return cls(prior, posteriors)

    @staticmethod
    def _get_mean(param):
        return NormalDiagonalCovariance._get_mean(param)

    @staticmethod
    def _get_cov(param):
        return NormalDiagonalCovariance._get_cov(param)

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)


class NormalSetFullCovariance(NormalSet):
    '''Set of Normal models with full covariance matrix.'''

    @classmethod
    def create(cls, mean, variance, size, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        cov = torch.diag(variance)
        scale = prior_strength
        dof = prior_strength + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posteriors = [
            NormalWishartPrior(
                mean + noise_std * torch.randn(len(mean), dtype=dtype,
                                                device=device),
                scale, scale_matrix, dof
            ) for _ in range(size)
        ]
        return cls(prior, posteriors)

    @staticmethod
    def _get_mean(param):
        return NormalFullCovariance._get_mean(param)

    @staticmethod
    def _get_cov(param):
        return NormalFullCovariance._get_cov(param)

    @staticmethod
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)


class NormalSetSharedIsotropicCovariance(BayesianModelSet):
    '''Set of Normal density models with a shared isotropic covariance
    matrix.
    '''

    @classmethod
    def create(cls, mean, variance, size, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        scales = torch.ones(size, dtype=dtype, device=device)
        scales *= prior_strength
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate = torch.tensor(prior_strength * variance.sum(), dtype=dtype,
                            device=device)
        p_means = mean + torch.zeros(size, len(mean), dtype=dtype, device=device)
        means = mean +  noise_std * torch.randn(size, len(mean), dtype=dtype,
                                                device=device)
        prior = JointIsotropicNormalGammaPrior(p_means, scales, shape, rate)
        posterior = JointIsotropicNormalGammaPrior(means, scales, shape, rate)
        return cls(prior, posterior)

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_precision = BayesianParameter(prior, posterior)

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_precision.expected_value(concatenated=False)
        cov = (1 / (-2 * np1)) * torch.eye(np2.shape[1], dtype=np1.dtype,
                                           device=np1.device)
        mean = cov @ np2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return self.means_precision.posterior.ncomp

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_precision.expected_value(concatenated=False)
        return torch.cat([np1.view(-1), np4.view(-1)]), \
            torch.cat([np2, np3.view(-1, 1)], dim=1).view(len(self), -1)

    def mean_field_factorization(self):
        return [[self.means_precision]]

    def sufficient_statistics(self, data):
        dtype, device = data.dtype, data.device
        padding = torch.ones(len(data), 1, dtype=dtype, device=device)
        s_stats1 = torch.cat([(data**2).sum(dim=1).view(-1, 1), padding], dim=1)
        s_stats2 = torch.cat([data, padding], dim=1)
        return s_stats1, s_stats2

    def forward(self, s_stats):
        s_stats1, s_stats2 = s_stats
        feadim = s_stats2.shape[1] - 1
        params = self._expected_nparams()
        retval = (s_stats1 @ params[0])[:, None] + s_stats2 @ params[1].t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        s_stats1, s_stats2, weights = *s_stats, parent_msg
        dtype, device = s_stats2.dtype, s_stats2.device
        acc_stats = torch.cat([
            s_stats1[:, 0].sum(dim=0).view(1),
            (weights.t() @ s_stats2[:, :-1]).view(-1),
            weights.sum(dim=0),
            torch.tensor(len(s_stats1), dtype=dtype, device=device).view(1)
        ])
        return {self.means_precision: acc_stats}


class NormalSetSharedDiagonalCovariance(BayesianModelSet):
    '''Set of Normal density models with a shared full covariance
    matrix.
    '''

    @classmethod
    def create(cls, mean, variance, size, prior_strength, noise_std):
        dtype, device = mean.dtype, mean.device
        scales = torch.ones(size, len(mean), dtype=dtype, device=device)
        scales *= prior_strength
        shape = torch.ones_like(variance) * prior_strength
        rate = variance * prior_strength
        p_means = mean + torch.zeros_like(scales, dtype=dtype, device=device)
        means = mean +  noise_std * torch.randn(size, len(mean), dtype=dtype,
                                                device=device)
        prior = JointNormalGammaPrior(p_means, scales, shape, rate)
        posterior = JointNormalGammaPrior(means, scales, shape, rate)
        return cls(prior, posterior)

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_precision = BayesianParameter(prior, posterior)

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_precision.expected_value(concatenated=False)

        cov = 1 / (-2 * np1)
        mean = cov * np2[key]
        return NormalSetElement(mean=mean, cov=torch.diag(cov))

    def __len__(self):
        return self.means_precision.posterior.ncomp

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_precision.expected_value(concatenated=False)
        return torch.cat([np1.view(-1), np4.view(-1)]), \
            torch.cat([np2, np3], dim=1)

    def mean_field_factorization(self):
        return [[self.means_precision]]

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = torch.cat([data ** 2, torch.ones_like(data)], dim=1)
        s_stats2 = torch.cat([data, torch.ones_like(data)], dim=1)
        return s_stats1, s_stats2

    def forward(self, s_stats):
        s_stats1, s_stats2 = s_stats
        feadim = s_stats1.size(1) // 2
        params = self._expected_nparams()
        retval = (s_stats1 @ params[0])[:, None] + s_stats2 @ params[1].t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        s_stats1, s_stats2, weights = *s_stats, parent_msg
        feadim = s_stats1.size(1) // 2

        acc_stats1 = s_stats1[:, :feadim].sum(dim=0)
        epsilon = torch.tensor(1e-2, dtype=s_stats1.dtype,
                               device=s_stats1.device)
        acc_stats1 = torch.where(acc_stats1 > epsilon, acc_stats1, epsilon)
        acc_stats = torch.cat([
            acc_stats1,
            (weights.t() @ s_stats2[:, :feadim]).view(-1),
            (weights.t() @ s_stats2[:, feadim:]).view(-1),
            len(s_stats1) * torch.ones(feadim, dtype=s_stats1.dtype,
                                       device=s_stats1.device)
        ])
        return {self.means_precision: acc_stats}


class NormalSetSharedFullCovariance(BayesianModelSet):
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


__all__ = [
    'NormalSetIsotropicCovariance',
    'NormalSetDiagonalCovariance',
    'NormalSetFullCovariance',
    'NormalSetSharedDiagonalCovariance',
    'NormalSetSharedFullCovariance'
]
