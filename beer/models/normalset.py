
'''Set of Normal densities with prior over the mean and
covariance matrix.

'''

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


class NormalIsotropicCovarianceSet(BayesianModelSet):
    '''Set of Normal models with isotropic covariance matrix.'''

    def __init__(self, prior, posteriors):
        super().__init__()
        self.normals = BayesianParameterSet([
            BayesianParameter(prior, post)
            for post in posteriors
        ])

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.mean_precision.expected_value(concatenated=False)
        return np2 / (-2 * np1)
        return NormalSetElement(mean=self._components[key].mean,
                                cov=self._components[key].cov)

    def __len__(self):
        return len(self.normals)

    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value()[None]
                          for param in self.normals], dim=0)

    @staticmethod
    def sufficient_statistics(data):
        return NormalIsotropicCovariance.sufficient_statistics(data)

    def forward(self, s_stats):
        feadim = s_stats.size(1) - 3
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        return dict(zip(self.parameters, weights.t() @ s_stats))


class NormalDiagonalCovarianceSet(BayesianModelSet):
    '''Set of Normal models with diagonal covariance matrix.'''

    def __init__(self, prior, posteriors):
        super().__init__()
        self._components = [
            NormalDiagonalCovariance(prior, post) for post in posteriors
        ]
        self._parameters = BayesianParameterSet([
            BayesianParameter(comp.parameters[0].prior,
                              comp.parameters[0].posterior)
            for comp in self._components
        ])

    def __getitem__(self, key):
        return NormalSetElement(mean=self._components[key].mean,
                                cov=self._components[key].cov)

    def __len__(self):
        return len(self._components)


    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value()[None]
                          for param in self.parameters], dim=0)

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)

    def forward(self, s_stats):
        feadim = .25 * s_stats.size(1)
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        return dict(zip(self.parameters, weights.t() @ s_stats))


class NormalFullCovarianceSet(BayesianModelSet):
    '''Set Normal models with full covariance matrix.'''

    def __init__(self, prior, posteriors):
        super().__init__()
        self._components = [
            NormalFullCovariance(prior, post) for post in posteriors
        ]
        self._parameters = BayesianParameterSet([
            BayesianParameter(comp.parameters[0].prior,
                              comp.parameters[0].posterior)
            for comp in self._components
        ])

    def __getitem__(self, key):
        return NormalSetElement(mean=self._components[key].mean,
                                cov=self._components[key].cov)

    def __len__(self):
        return len(self._components)

    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value()[None]
                          for param in self.parameters], dim=0)

    @staticmethod
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)

    def forward(self, s_stats):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - s_stats.size(1))))
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        return dict(zip(self.parameters, weights.t() @ s_stats))


class NormalSetSharedIsotropicCovariance(BayesianModelSet):
    '''Set of Normal density models with a shared isotropic covariance
    matrix.
    '''

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_prec_param.expected_value(concatenated=False)

        cov = (1 / (-2 * np1)) * torch.eye(np2.shape[1], dtype=np1.dtype,
                                           device=np1.device)
        mean = cov @ np2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return self.means_prec_param.posterior.ncomp

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.expected_value(concatenated=False)
        return torch.cat([np1.view(-1), np4.view(-1)]), \
            torch.cat([np2, np3.view(-1, 1)], dim=1).view(len(self), -1)

    def expected_natural_params_as_matrix(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.expected_value(concatenated=False)
        ones = torch.ones_like(np2)
        return torch.cat([ones * np1, np2, ones * np3[:, None] ,
                          ones * np4], dim=1)

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
        return {self.means_prec_param: acc_stats}


class NormalSetSharedDiagonalCovariance(BayesianModelSet):
    '''Set of Normal density models with a shared full covariance
    matrix.
    '''

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_prec_param.expected_value(concatenated=False)

        cov = 1 / (-2 * np1)
        mean = cov * np2[key]
        return NormalSetElement(mean=mean, cov=torch.diag(cov))

    def __len__(self):
        return self.means_prec_param.posterior.ncomp

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.expected_value(concatenated=False)
        return torch.cat([np1.view(-1), np4.view(-1)]), \
            torch.cat([np2, np3], dim=1)

    def expected_natural_params_as_matrix(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.expected_value(concatenated=False)
        ones = torch.ones_like(np2)
        return torch.cat([ones * np1, np2, np3, ones * np4], dim=1)

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = torch.cat([data**2, torch.ones_like(data)], dim=1)
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
        acc_stats = torch.cat([
            s_stats1[:, :feadim].sum(dim=0),
            (weights.t() @ s_stats2[:, :feadim]).view(-1),
            (weights.t() @ s_stats2[:, feadim:]).view(-1),
            len(s_stats1) * torch.ones(feadim, dtype=s_stats1.dtype,
                                       device=s_stats1.device)
        ])
        return {self.means_prec_param: acc_stats}

    def expected_natural_params_from_resps(self, resps):
        matrix = self.expected_natural_params_as_matrix()
        return resps @ matrix


class NormalSetSharedFullCovariance(BayesianModelSet):
    '''Set of Normal density models with a  shared covariance matrix.'''

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_prec_param.expected_value(concatenated=False)
        cov = torch.inverse(-2 * np1)
        mean = cov @ np2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return self.means_prec_param.posterior.ncomp

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.expected_value(concatenated=False)
        return np1.view(-1), \
            torch.cat([np2, np3[:, None]], dim=1), np4

    def expected_natural_params_as_matrix(self):
        dim = self.means_prec_param.posterior.dim
        np1, np2, np3, np4 = \
            self.means_prec_param.expected_value(concatenated=False)
        ones1 = torch.ones(self._ncomp, dim ** 2, dtype=np1.dtype,
                           device=np1.device)
        ones2 = torch.ones(self._ncomp, 1, dtype=np1.dtype, device=np2.device)
        return torch.cat([ones1 * np1.view(-1)[None, :],
                          np2, np3.view(-1, 1), ones2 * np4], dim=1)

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
        return {self.means_prec_param: acc_stats}


def create(model_conf, mean, variance, create_model_handle):
    dim = len(mean)
    dtype, device = mean.dtype, mean.device
    n_element = model_conf['size']
    covariance_type = model_conf['covariance']
    shared_cov = model_conf['shared_covariance']
    noise_std = model_conf['noise_std']
    prior_strength = model_conf['prior_strength']
    if covariance_type == 'isotropic':
        if shared_cov:
            scales = torch.ones(n_element, dtype=dtype, device=device)
            scales *= prior_strength
            shape = torch.tensor(prior_strength, dtype=dtype, device=device)
            rate = torch.tensor(prior_strength * variance.sum(), dtype=dtype,
                                device=device)
            p_means = mean + torch.zeros(n_element, dim, dtype=dtype,
                                         device=device)
            means = mean +  noise_std * torch.randn(n_element, dim, dtype=dtype,
                                                    device=device)
            prior = JointIsotropicNormalGammaPrior(p_means, scales, shape, rate)
            posterior = JointIsotropicNormalGammaPrior(means, scales, shape, rate)
            return NormalSetSharedIsotropicCovariance(prior, posterior)
        else:
            scale = torch.tensor(prior_strength, dtype=dtype, device=device)
            shape = torch.tensor(prior_strength, dtype=dtype, device=device)
            rate = torch.tensor(prior_strength * variance.sum(), dtype=dtype,
                                device=device)
            prior = IsotropicNormalGammaPrior(mean, scale, shape, rate)
            rand_means = noise_std * torch.randn(n_element, dim, dtype=dtype,
                                                 device=device) + mean
            posteriors = [IsotropicNormalGammaPrior(rand_means[i], scale, shape,
                                                    rate)
                          for i in range(n_element)]
            return NormalIsotropicCovarianceSet(prior, posteriors)
    elif covariance_type == 'diagonal':
        if shared_cov:
            scales = torch.ones(n_element, dim, dtype=dtype, device=device)
            scales *= prior_strength
            shape = torch.ones_like(variance) * prior_strength
            rate = variance * prior_strength
            p_means = mean + torch.zeros_like(scales, dtype=dtype, device=device)
            means = mean +  noise_std * torch.randn(n_element, dim, dtype=dtype,
                                                    device=device)
            prior = JointNormalGammaPrior(p_means, scales, shape, rate)
            posterior = JointNormalGammaPrior(means, scales, shape, rate)
            return NormalSetSharedDiagonalCovariance(prior, posterior)
        else:
            scale = torch.ones_like(mean) * prior_strength
            shape = torch.ones_like(mean) * prior_strength
            rate = prior_strength * variance
            prior = NormalGammaPrior(mean, scale, shape, rate)
            rand_means = noise_std * torch.randn(n_element, dim, dtype=dtype,
                                                 device=device) + mean
            posteriors = [NormalGammaPrior(rand_means[i], scale, shape, rate)
                          for i in range(n_element)]
            return NormalDiagonalCovarianceSet(prior, posteriors)
    elif covariance_type == 'full':
        cov = torch.diag(variance)
        if shared_cov:
            scales = torch.ones(n_element, dtype=dtype, device=device)
            scales *= prior_strength
            dof = prior_strength + dim - 1
            scale_matrix = torch.inverse(cov *  dof)
            p_means = mean + torch.zeros(n_element, dim, dtype=mean.dtype,
                                        device=mean.device)
            means = mean + noise_std * torch.randn(n_element, dim, dtype=dtype,
                                                   device=device)
            prior = JointNormalWishartPrior(p_means, scales, scale_matrix, dof)
            posteriors = JointNormalWishartPrior(means, scales, scale_matrix, dof)
            return NormalSetSharedFullCovariance(prior, posteriors)
        else:
            scale = prior_strength
            dof = prior_strength + len(mean) - 1
            scale_matrix = torch.inverse(cov *  dof)
            prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
            posteriors = [
                NormalWishartPrior(
                    mean + noise_std * torch.randn(len(mean), dtype=dtype,
                                                   device=device),
                    scale, scale_matrix, dof
                ) for _ in range(n_element)
            ]
            return NormalFullCovarianceSet(prior, posteriors)
    else:
        raise ValueError('Unknown covariance type: {}'.format(covariance_type))


__all__ = [
    'NormalIsotropicCovarianceSet',
    'NormalDiagonalCovarianceSet',
    'NormalFullCovarianceSet',
    'NormalSetSharedDiagonalCovariance',
    'NormalSetSharedFullCovariance'
]
