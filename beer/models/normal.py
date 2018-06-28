
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
    '''Bayesian Normal density with diagonal covariance matrix.

    Attributes:
        mean (``torch.Tensor``): Expected mean.
        cov (``torch.Tensor``): Expected (diagonal) covariance matrix.

    Example:
        >>> # Create a Normal with zero mean and identity covariamce
        >>> # matrix.
        >>> mean = torch.zeros(2)
        >>> diav_cov = torch.ones(2)
        >>> normal = beer.NormalDiagonalCovariance.create(mean, diag_cov)
        >>> normal.mean
        tensor([ 0.,  0.])
        >>> model.cov
        tensor([[ 1.,  0.],
                [ 0.,  1.]])

    '''

    def __init__(self, prior, posterior):
        '''
        Args:
            prior (:any:`beer.NormalGammaPrior`): Prior over the mean
                and the diagonal of the precision matrix
            posterior (:any:`beer.NormalGammaPrior`): Posterior over the
                mean and the diagonal of the precision matrix.
        '''
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, mean, diag_cov, pseudo_counts=1.):
        '''Create a :any:`NormalDiagonalCovariance`.

        Args:
            mean (``torch.Tensor``): Mean of the Normal to create.
            diag_cov (``torch.Tensor``): Diagonal of the covariance
                matrix of the Normal to create.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.

        Returns:
            :any:`NormalDiagonalCovariance`
        '''
        scale = torch.ones_like(mean) * pseudo_counts
        shape = torch.ones_like(mean) * pseudo_counts
        rate = pseudo_counts * diag_cov
        prior = NormalGammaPrior(mean, scale, shape, rate)
        posterior = NormalGammaPrior(mean, scale, shape, rate)
        return cls(prior, posterior)

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

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([data ** 2, data, torch.ones_like(data),
                          torch.ones_like(data)], dim=-1)

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

    def forward(self, s_stats):
        feadim = .25 * s_stats.size(1)
        exp_llh = s_stats @ self.mean_prec_param.expected_value()
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

    def expected_natural_params(self, mean, var, nsamples=1):
        '''Interface for the VAE model. Returns the expected value of the
        natural params of the latent model given the per-frame means
        and variances.

        Args:
            mean (``torch.Tensor``): Per-frame mean of the posterior
                distribution.
            var (``torch.Tensor``): Per-frame variance of the posterior
                distribution.
            nsamples (int): Number of samples to estimate the
                natural parameters.

        Returns:
            (``torch.Tensor``): Expected value of the natural parameters.

        '''
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)
        nparams = self.mean_prec_param.expected_value()
        ones = torch.ones(s_stats.size(0), nparams.size(0), dtype=s_stats.dtype,
                          device=s_stats.device)
        return ones * nparams, s_stats


class NormalFullCovariance(BayesianModel):
    '''Bayesian Normal distribution with a full covariance matrix.

    Attributes:
        mean (``torch.Tensor``): Expected mean.
        cov (``torch.Tensor``): Expected covariance matrix.

    Example:
        >>> # Create a Normal with zero mean and identity covariamce
        >>> # matrix.
        >>> mean = torch.zeros(2)
        >>> cov = torch.eye(2)
        >>> normal = beer.NormalFullCovariance.create(mean, cov)
        >>> normal.mean
        tensor([ 0.,  0.])
        >>> model.cov
        tensor([[ 1.,  0.],
                [ 0.,  1.]])

    '''

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, mean, cov, pseudo_counts=1.):
        '''Create a :any:`NormalFullCovariance`.

        Args:
            mean (``torch.Tensor``): Mean of the Normal to create.
            cov (``torch.Tensor``): Covariance matrix of the Normal to
                create.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.

        Returns:
            :any:`NormalFullCovariance`
        '''
        scale = pseudo_counts
        dof = pseudo_counts + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posterior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        return cls(prior, posterior)

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

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([
            (data[:, :, None] * data[:, None, :]).view(len(data), -1),
            data, torch.ones(data.size(0), 1, dtype=data.dtype,
                             device=data.device),
            torch.ones(data.size(0), 1, dtype=data.dtype, device=data.device)
        ], dim=-1)

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

    def forward(self, s_stats):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - s_stats.size(1))))
        exp_llh = s_stats @ self.mean_prec_param.expected_value()
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}


#######################################################################
# NormalSet model
#######################################################################

NormalSetElement = namedtuple('NormalSetElement', ['mean', 'cov'])

class NormalDiagonalCovarianceSet(BayesianModelSet):
    '''Set of Normal density models with diagonal covariance.

    Note:
        All the Normal models of the set will share the same prior
        distribution.

    Example:
        >>> # Create a set of Normal densities.
        >>> mean = torch.zeros(2)
        >>> diav_cov = torch.ones(2)
        >>> normalset = beer.NormalDiagonalCovarianceSet.create(mean, diag_cov, 3)
        >>> len(normalset)
        3

    '''

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
        '''Create a :any:`NormalDiagonalCovarianceSet`.

        Args:
            mean (``torch.Tensor``): Mean of the Normal to create.
            diag_cov (``torch.Tensor``): Diagonal of the covariance
                matrix of the Normal to create.
            ncomp (int): Number of component in the set.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.
            noise_std (float): Standard deviation of the noise when
                initializing the mean of the posterior distribution.

        Returns:
            :any:`NormalDiagonalCovarianceSet`
        '''
        scale = torch.ones_like(mean) * pseudo_counts
        shape = torch.ones_like(mean) * pseudo_counts
        rate = pseudo_counts * diag_cov
        prior = NormalGammaPrior(mean, scale, shape, rate)
        posteriors = [
            NormalGammaPrior(
                mean + noise_std * torch.randn(len(mean), dtype=mean.dtype,
                                               device=mean.device),
                scale, shape, rate
            ) for _ in range(ncomp)
        ]
        return cls(prior, posteriors)

    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value()[None]
                          for param in self.parameters], dim=0)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)

    def float(self):
        new_prior = self._components[0].mean_prec_param.prior.float()
        new_posts = [comp.mean_prec_param.posterior.float()
                     for comp in self._components]
        return self.__class__(new_prior, new_posts)

    def double(self):
        new_prior = self._components[0].mean_prec_param.prior.double()
        new_posts = [comp.mean_prec_param.posterior.double()
                     for comp in self._components]
        return self.__class__(new_prior, new_posts)

    def to(self, device):
        new_prior = self._components[0].mean_prec_param.prior.to(device)
        new_posts = [comp.mean_prec_param.posterior.to(device)
                     for comp in self._components]
        return self.__class__(new_prior, new_posts)

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

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        return NormalSetElement(mean=self._components[key].mean,
                                cov=self._components[key].cov)

    def __len__(self):
        return len(self._components)

    def expected_natural_params_from_resps(self, resps):
        matrix = self.expected_natural_params_as_matrix()
        return resps @ matrix

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            mean, var)



class NormalFullCovarianceSet(BayesianModelSet):
    '''Set Normal density models with full covariance matrix.

    Note:
        All the Normal models of the set will share the same prior
        distribution.

    Example:
        >>> # Create a set of Normal densities.
        >>> mean = torch.zeros(2)
        >>> cov = torch.eye(2)
        >>> normalset = beer.NormalFullCovarianceSet.create(mean, cov, 3)
        >>> len(normalset)
        3

    '''

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
        '''Create a :any:`NormalFullCovarianceSet`.

        Args:
            mean (``torch.Tensor``): Mean of the Normal to create.
            cov (``torch.Tensor``): Covariance matrix of the Normal to
                create.
            ncomp (int): Number of component in the set.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.
            noise_std (float): Standard deviation of the noise when
                initializing the mean of the posterior distribution.

        Returns:
            :any:`NormalFullCovarianceSet`
        '''
        scale = pseudo_counts
        dof = pseudo_counts + len(mean) - 1
        scale_matrix = torch.inverse(cov *  dof)
        prior = NormalWishartPrior(mean, scale, scale_matrix, dof)
        posteriors = [
            NormalWishartPrior(
                mean + noise_std * torch.randn(len(mean), dtype=mean.dtype,
                                               device=mean.device),
                scale, scale_matrix, dof
            ) for _ in range(ncomp)
        ]
        return cls(prior, posteriors)

    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value()[None]
                          for param in self.parameters], dim=0)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)

    def float(self):
        new_prior = self._components[0].mean_prec_param.prior.float()
        new_posts = [comp.mean_prec_param.posterior.float()
                     for comp in self._components]
        return self.__class__(new_prior, new_posts)

    def double(self):
        new_prior = self._components[0].mean_prec_param.prior.double()
        new_posts = [comp.mean_prec_param.posterior.double()
                     for comp in self._components]
        return self.__class__(new_prior, new_posts)

    def to(self, device):
        new_prior = self._components[0].mean_prec_param.prior.to(device)
        new_posts = [comp.mean_prec_param.posterior.to(device)
                     for comp in self._components]
        return self.__class__(new_prior, new_posts)

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

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        return NormalSetElement(mean=self._components[key].mean,
                                cov=self._components[key].cov)

    def __len__(self):
        return len(self._components)

    def expected_natural_params_from_resps(self, resps):
        matrix = self.expected_natural_params_as_matrix()
        return resps @ matrix


class NormalSetSharedDiagonalCovariance(BayesianModelSet):
    '''Set of Normal density models with a global shared full
    covariance matrix.

    Example:
        >>> # Create a set of Normal densities.
        >>> mean = torch.zeros(2)
        >>> diag_cov = torch.ones(2)
        >>> normalset = beer.NormalSetSharedDiagonalCovariance.create(mean, diag_cov, 3)
        >>> len(normalset)
        3

    '''

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, mean, diag_cov, ncomp, pseudo_counts=1., noise_std=0.):
        '''Create a :any:`NormalSetSharedDiagonalCovariance`.

        Args:
            mean (``torch.Tensor``): Mean of the Normal to create.
            diag_cov (``torch.Tensor``): Diagonal of the covariance
                matrix of the Normal to create.
            ncomp (int): Number of component in the set.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.
            noise_std (float): Standard deviation of the noise when
                initializing the mean of the posterior distribution.

        Returns:
            :any:`NormalSetSharedDiagonalCovariance`
        '''
        dim = len(mean)
        scales = torch.ones(ncomp, dim, dtype=mean.dtype,
                            device=mean.device) * pseudo_counts
        shape = torch.ones_like(diag_cov) * pseudo_counts
        rate = diag_cov * pseudo_counts
        p_means = mean + torch.zeros_like(scales, dtype=mean.dtype,
                                          device=mean.device,)
        means = mean +  noise_std * torch.randn(ncomp, dim, dtype=mean.dtype,
                                                device=mean.device)
        prior = JointNormalGammaPrior(p_means, scales, shape, rate)
        posterior = JointNormalGammaPrior(means, scales, shape, rate)
        return cls(prior, posterior)

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

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = torch.cat([data**2, torch.ones_like(data)], dim=1)
        s_stats2 = torch.cat([data, torch.ones_like(data)], dim=1)
        return s_stats1, s_stats2

    def float(self):
        return self.__class__(
            self.means_prec_param.prior.float(),
            self.means_prec_param.posterior.float()
        )

    def double(self):
        return self.__class__(
            self.means_prec_param.prior.double(),
            self.means_prec_param.posterior.double()
        )

    def to(self, device):
        return self.__class__(
            self.means_prec_param.prior.to(device),
            self.means_prec_param.posterior.to(device)
        )

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

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_prec_param.expected_value(concatenated=False)

        cov = 1 / (-2 * np1)
        mean = cov * np2[key]
        return NormalSetElement(mean=mean, cov=torch.diag(cov))

    def __len__(self):
        return self.means_prec_param.posterior.ncomp

    def expected_natural_params_from_resps(self, resps):
        matrix = self.expected_natural_params_as_matrix()
        return resps @ matrix

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        s_stats1 = torch.cat([mean ** 2 + var, torch.ones_like(mean)], dim=1)
        s_stats2 = torch.cat([mean, torch.ones_like(mean)], dim=1)
        return s_stats1, s_stats2


class NormalSetSharedFullCovariance(BayesianModelSet):
    '''Set of Normal density models with a globale shared full
    covariance matrix.

    Example:
        >>> # Create a set of Normal densities.
        >>> mean = torch.zeros(2)
        >>> cov = torch.eye(2)
        >>> normalset = beer.NormalSetSharedFullCovariance.create(mean, cov, 3)
        >>> len(normalset)
        3

    '''

    def __init__(self, prior, posterior):
        super().__init__()
        self._ncomp = prior.ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    @classmethod
    def create(cls, mean, cov, ncomp, pseudo_counts=1., noise_std=0.):
        '''Create a :any:`NormalSetSharedFullCovariance`.

        Args:
            mean (``torch.Tensor``): Mean of the Normal to create.
            cov (``torch.Tensor``): Covariance matrix of the Normal to
                create.
            ncomp (int): Number of component in the set.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.
            noise_std (float): Standard deviation of the noise when
                initializing the mean of the posterior distribution.

        Returns:
            :any:`NormalSetSharedFullCovariance`
        '''
        dim = len(mean)
        scales = torch.ones(ncomp, dtype=mean.dtype,
                            device=mean.device) * pseudo_counts
        dof = pseudo_counts + dim - 1
        scale_matrix = torch.inverse(cov *  dof)
        p_means = mean + torch.zeros(ncomp, dim, dtype=mean.dtype,
                                     device=mean.device)
        means = mean + noise_std * torch.randn(ncomp, dim, dtype=mean.dtype,
                                              device=mean.device)
        prior = JointNormalWishartPrior(p_means, scales, scale_matrix, dof)
        posteriors = JointNormalWishartPrior(means, scales, scale_matrix, dof)
        return cls(prior, posteriors)

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

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = (data[:, :, None] * data[:, None, :]).view(len(data), -1)
        s_stats2 = torch.cat([data, torch.ones(data.size(0), 1,
                                               dtype=data.dtype,
                                               device=data.device)], dim=1)
        return s_stats1, s_stats2

    def float(self):
        return self.__class__(
            self.means_prec_param.prior.float(),
            self.means_prec_param.posterior.float()
        )

    def double(self):
        return self.__class__(
            self.means_prec_param.prior.double(),
            self.means_prec_param.posterior.double()
        )

    def to(self, device):
        return self.__class__(
            self.means_prec_param.prior.to(device),
            self.means_prec_param.posterior.to(device)
        )

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

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        np1, np2, _, _ = \
            self.means_prec_param.expected_value(concatenated=False)
        cov = torch.inverse(-2 * np1)
        mean = cov @ np2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return self.means_prec_param.posterior.ncomp

    def expected_natural_params_from_resps(self, resps):
        matrix = self.expected_natural_params_as_matrix()
        return resps @ matrix


__all__ = ['NormalDiagonalCovariance', 'NormalFullCovariance',
           'NormalDiagonalCovarianceSet', 'NormalFullCovarianceSet',
           'NormalSetSharedDiagonalCovariance', 'NormalSetSharedFullCovariance']
