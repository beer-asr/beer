
'''Bayesian Normal distribution with prior over the mean and
covariance matrix.


 Normal model
 ------------
   The ``Normal`` model is very simple model that fits a data with
   a Normal density. Practically, the Normal class is just an
   interface. It has 2 concrete implementations, one with diagonal
   covariance matrix and the other one with full covariance matrix.


NormalSet
---------
   The ``NormalSet`` object is not a model but rather a component of
   more comple model (e.g. GMM). For example, it allows to have a set of
   Normal densities to with a shared prior distribution for ex.


'''

import abc
from collections import namedtuple
import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianModelSet

from ..expfamilyprior import NormalGammaPrior
from ..expfamilyprior import JointNormalGammaPrior
from ..expfamilyprior import NormalWishartPrior
from ..expfamilyprior import JointNormalWishartPrior


class NormalDiagonalCovariance(BayesianModel):
    '''Bayesian Normal distribution with diagonal covariance matrix.'''

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, mean, diag_cov, pseudo_counts=1.):
        scale = torch.ones_like(mean) * pseudo_counts
        shape = torch.ones_like(mean) * pseudo_counts
        rate = pseudo_counts * diag_cov
        prior = NormalGammaPrior(mean, scale, shape, rate)
        posterior = NormalGammaPrior(mean, scale, shape, rate)
        return cls(prior, posterior)

    @property
    def mean(self):
        np1, np2, _, _ = \
            self.mean_prec_param.posterior.split_sufficient_statistics(
                self.mean_prec_param.expected_value
            )
        return np2 / (-2 * np1)

    @property
    def cov(self):
        np1, _, _, _ = \
            self.mean_prec_param.posterior.split_sufficient_statistics(
                self.mean_prec_param.expected_value
            )
        return torch.diag(1/(-2 * np1))

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([data ** 2, data, torch.ones_like(data),
                          torch.ones_like(data)], dim=-1)

    def forward(self, s_stats, latent_variables=None):
        feadim = .25 * s_stats.size(1)
        exp_llh = s_stats @ self.mean_prec_param.expected_value
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([(mean ** 2) + var, mean, torch.ones_like(mean),
                          torch.ones_like(mean)], dim=-1)

    # pylint: disable=W0613
    # Unused arguments (labels and nsamples).
    def expected_natural_params(self, mean, var, labels=None, nsamples=1):
        '''Interface for the VAE model. Returns the expected value of the
        natural params of the latent model given the per-frame means
        and variances.

        Args:
            mean (Tensor): Per-frame mean of the posterior distribution.
            var (Tensor): Per-frame variance of the posterior
                distribution.
            labels (Tensor): Frame labelling (if any).
            nsamples (int): Number of samples to estimate the
                natural parameters.

        Returns:
            (Tensor): Expected value of the natural parameters.

        '''
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)
        nparams = self.mean_prec_param.expected_value
        ones = torch.ones(s_stats.size(0), nparams.size(0)).type(s_stats.type())
        return ones * nparams


class NormalFullCovariance(BayesianModel):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, mean, cov, pseudo_counts=1.):
        scale = pseudo_counts
        dof = pseudo_counts + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posterior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        return cls(prior, posterior)

    @property
    def mean(self):
        np1, np2, _, _ = \
            self.mean_prec_param.posterior.split_sufficient_statistics(
                self.mean_prec_param.expected_value
            )
        return torch.inverse(-2 * np1) @ np2

    @property
    def cov(self):
        np1, _, _, _ = \
            self.mean_prec_param.posterior.split_sufficient_statistics(
                self.mean_prec_param.expected_value
            )
        return torch.inverse(-2 * np1)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([
            (data[:, :, None] * data[:, None, :]).view(len(data), -1),
            data, torch.ones(data.size(0), 1).type(data.type()),
            torch.ones(data.size(0), 1).type(data.type())
        ], dim=-1)

    def forward(self, s_stats, labels=None):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - s_stats.size(1))))
        exp_llh = s_stats @ self.mean_prec_param.expected_value
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}


#######################################################################
# NormalSet model
#######################################################################

NormalSetElement = namedtuple('NormalSetElement', ['mean', 'cov'])


class NormalDiagonalCovarianceSet(BayesianModelSet):
    'Set Normal density models with diagonal covariance.'

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

    @classmethod
    def create(cls, mean, diag_cov, ncomp, pseudo_counts=1., noise_std=0.):
        scale = torch.ones_like(mean) * pseudo_counts
        shape = torch.ones_like(mean) * pseudo_counts
        rate = pseudo_counts * diag_cov
        prior = NormalGammaPrior(mean, scale, shape, rate)
        posteriors = [
            NormalGammaPrior(mean + noise_std * torch.randn(len(mean)),
                             scale, shape, rate)
            for _ in range(ncomp)
        ]
        return cls(prior, posteriors)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)

    def forward(self, s_stats, latent_variables=None):
        feadim = .25 * s_stats.size(1)
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        return dict(zip(self.parameters, weights.t() @ s_stats))

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        return NormalSetElement(mean=self._components[key].mean,
                                cov=self._components[key].cov)

    def __len__(self):
        return len(self._components)

    # pylint: disable=C0103
    # Invalid method name.
    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value[None]
                          for param in self.parameters], dim=0)

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            mean, var)


class NormalFullCovarianceSet(BayesianModelSet):
    'Set Normal density models with full covariance.'

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

    @classmethod
    def create(cls, mean, cov, ncomp, pseudo_counts=1., noise_std=0.):
        scale = pseudo_counts
        dof = pseudo_counts + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posteriors = [
            NormalWishartPrior(mean + noise_std * torch.randn(len(mean)),
                               scale, scale_matrix, dof)
            for _ in range(ncomp)
        ]
        return cls(prior, posteriors)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)

    def forward(self, s_stats, latent_variables=None):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - s_stats.size(1))))
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        return dict(zip(self.parameters, weights.t() @ s_stats))

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        return NormalSetElement(mean=self._components[key].mean,
                                cov=self._components[key].cov)

    def __len__(self):
        return len(self._components)

    # pylint: disable=C0103
    # Invalid method name.
    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value[None]
                          for param in self.parameters], dim=0)


class NormalSetSharedDiagonalCovariance(BayesianModelSet):
    '''Set of Normal density models with a globale shared full
    covariance matrix.

    '''

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, means, diag_cov, pseudo_counts=1., noise_std=0.):
        scales = torch.ones_like(means) * pseudo_counts
        shape = torch.ones_like(diag_cov) * pseudo_counts
        rate = diag_cov * pseudo_counts
        prior = JointNormalGammaPrior(means, scales, shape, rate)
        posterior = JointNormalGammaPrior(
            means + noise_std * torch.randn(*means.size()),
            scales, shape, rate
        )
        return cls(prior, posterior)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = torch.cat([data**2, torch.ones_like(data)], dim=1)
        s_stats2 = torch.cat([data, torch.ones_like(data)], dim=1)
        return s_stats1, s_stats2

    def forward(self, s_stats, latent_variables=None):
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
            len(s_stats1) * torch.ones(feadim).type(s_stats1.type())
        ])
        return {self.means_prec_param: acc_stats}

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_prec_param.posterior.split_sufficient_statistics(
                self.means_prec_param.expected_value
            )
        cov = 1 / (-2 * np1)
        mean = cov * np2[key]
        return NormalSetElement(mean=mean, cov=torch.diag(cov))

    def __len__(self):
        return self.means_prec_param.posterior.ncomp

    # pylint: disable=C0103
    # Invalid method name.
    def expected_natural_params_as_matrix(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.posterior.split_sufficient_statistics(
                self.means_prec_param.expected_value
            )
        ones = torch.ones_like(np2)
        return torch.cat([ones * np1, np2, np3, ones * np4], dim=1)

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.posterior.split_sufficient_statistics(
                self.means_prec_param.expected_value
            )
        return torch.cat([np1.view(-1), np4.view(-1)]), \
            torch.cat([np2, np3], dim=1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        s_stats1 = torch.cat([mean ** 2 + var, torch.ones_like(mean)], dim=1)
        s_stats2 = torch.cat([mean, torch.ones_like(mean)], dim=1)
        return s_stats1, s_stats2


class NormalSetSharedFullCovariance(BayesianModelSet):
    '''Set of Normal density models with a globale shared full
    covariance matrix.

    '''

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, means, cov, pseudo_counts=1., noise_std=0.):
        ncomp, dim = means.size()
        scales = torch.ones_like(means[:, 0]) * pseudo_counts
        dof = pseudo_counts + means.size(1) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = JointNormalWishartPrior(means, scales, scale_matrix, dof)
        posteriors = JointNormalWishartPrior(
            means + noise_std * torch.randn(ncomp, dim),
            scales, scale_matrix, dof
        )
        return cls(prior, posteriors)

    def _expected_nparams(self):
        np1, np2, np3, np4 = \
            self.means_prec_param.posterior.split_sufficient_statistics(
                self.means_prec_param.expected_value
            )
        return np1.view(-1), \
            torch.cat([np2, np3[:, None]], dim=1), np4

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = (data[:, :, None] * data[:, None, :]).view(len(data), -1)
        s_stats2 = torch.cat([data, torch.ones(data.size(0), 1).type(data.type())],
                             dim=1)
        return s_stats1, s_stats2

    def forward(self, s_stats, latent_variables=None):
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
            len(weights) * torch.ones(1).type(weights.type())
        ])
        return {self.means_prec_param: acc_stats}

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_prec_param.posterior.split_sufficient_statistics(
                self.means_prec_param.expected_value
            )
        cov = torch.inverse(-2 * np1)
        mean = cov @ np2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return self.means_prec_param.posterior.ncomp

    # pylint: disable=C0103
    def expected_natural_params_as_matrix(self):
        dim = self.means_prec_param.posterior.dim
        np1, np2, np3, np4 = \
            self.means_prec_param.posterior.split_sufficient_statistics(
                self.means_prec_param.expected_value
            )
        ones1 = torch.ones(self._ncomp, dim ** 2).type(np1.type())
        ones2 = torch.ones(self._ncomp, 1).type(np1.type())
        return torch.cat([ones1 * np1.view(-1)[None, :],
                          np2, np3.view(-1, 1), ones2 * np4], dim=1)
