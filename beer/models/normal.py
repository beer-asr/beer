
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
   more comple model (e.g. GMM). It allows to have a set of Normal
   densities to have a shared prior distribution.


'''

import abc
from collections import namedtuple
import math

import torch
import torch.autograd as ta

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .bayesmodel import BayesianModel
from ..expfamilyprior import NormalGammaPrior
from ..expfamilyprior import NormalWishartPrior
from ..expfamilyprior import kl_div
from ..expfamilyprior import _normalwishart_split_nparams
from ..expfamilyprior import _jointnormalwishart_split_nparams
from ..expfamilyprior import _jointnormalgamma_split_nparams


#######################################################################
# Normal model
#######################################################################

class Normal(BayesianModel, metaclass=abc.ABCMeta):
    'Abstract Base Class for the Normal distribution model.'

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        '''Compute the sufficient statistics of the data.

        Args:
            X (Tensor): Data.

        Returns:
            (Tensor): Sufficient statistics of the data.

        '''
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics_from_mean_var(mean, var):
        '''Compute the sufficient statistics of the data specified
        in term of a mean and variance for each data point.

        Args:
            mean (Tensor): Means for each point of the data.
            var (Tensor): Variances for each point (and
                dimension) of the data.

        Returns:
            (Tensor): Sufficient statistics of the data.

        '''
        NotImplemented

    @property
    @abc.abstractmethod
    def mean(self):
        'Expected value of the mean w.r.t. posterior distribution.'
        NotImplemented

    @property
    @abc.abstractmethod
    def cov(self):
        '''Expected value of the covariance matrix w.r.t posterior
         distribution.

        '''
        NotImplemented


class NormalDiagonalCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def sufficient_statistics(X):
        return torch.cat([X ** 2, X, torch.ones_like(X), torch.ones_like(X)],
                         dim=-1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([(mean ** 2) + var, mean, torch.ones_like(mean),
                          torch.ones_like(mean)], dim=-1)

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @property
    def mean(self):
        evalue = self.mean_prec_param.expected_value
        np1, np2, _, _ = evalue.view(4, -1)
        return np2 / (-2 * np1)

    @property
    def cov(self):
        evalue = self.mean_prec_param.expected_value
        np1, np2, _, _ = evalue.view(4, -1)
        return torch.diag(1/(-2 * np1))

    def expected_natural_params(self, means, vars):
        '''Interface for the VAE model. Returns the expected value of the
        natural params of the latent model given the per-frame means
        and variances.

        Args:
            means (Tensor): Per-frame means.
            vars (Tensor): Per-frame variances.

        Returns:
            (Tensor): Expected value of the natural parameters.

        '''
        return self.mean_prec_param.expected_value

    def forward(self, T, labels=None):
        feadim = .25 * T.size(1)
        exp_llh = T @ self.mean_prec_param.expected_value
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, T, parent_message=None):
        return {self.mean_prec_param: T.sum(dim=0)}


class NormalFullCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def sufficient_statistics(X):
        return torch.cat([(X[:, :, None] * X[:, None, :]).view(len(X), -1),
            X, torch.ones(X.size(0), 1).type(X.type()),
            torch.ones(X.size(0), 1).type(X.type())], dim=-1)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        raise NotImplementedError

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_prec_param = BayesianParameter(prior, posterior)

    @property
    def mean(self):
        evalue = self.mean_prec_param.expected_value
        np1, np2, _, _, _ = _normalwishart_split_nparams(evalue)
        return torch.inverse(-2 * np1) @ np2

    @property
    def cov(self):
        evalue = self.mean_prec_param.expected_value
        np1, _, _, _, _ = _normalwishart_split_nparams(evalue)
        return torch.inverse(-2 * np1)

    def forward(self, T, labels=None):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - T.size(1))))
        exp_natural_params = self.mean_prec_param.expected_value
        exp_llh = T @ self.mean_prec_param.expected_value
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, T, parent_message=None):
        return {self.mean_prec_param: T.sum(dim=0)}


#######################################################################
# NormalSet model
#######################################################################


NormalSetElement = namedtuple('NormalSetElement', ['mean', 'cov'])


class NormalSet(BayesianModel, metaclass=abc.ABCMeta):
    'Set Normal density models.'

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics_from_mean_var(mean, var):
        NotImplemented

    def __init__(self, components):
        super().__init__()
        self._components = components
        self._parameters = BayesianParameterSet([
            BayesianParameter(comp.parameters[0].prior,
                              comp.parameters[0].posterior)
            for comp in self._components
        ])

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('expected integer key')
        return NormalSetElement(mean=self._components[key].mean,
            cov=self._components[key].cov)

    def __len__(self):
        return len(self._components)

    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value[None]
            for param in self._parameters], dim=0)

    def accumulate(self, T, weights):
        return dict(zip(self.parameters, weights.t() @ T))


class NormalDiagonalCovarianceSet(NormalSet):
    'Set Normal density models with diagonal covariance.'

    def __init__(self, prior, posteriors):
        components = [
            NormalDiagonalCovariance(prior, post) for post in posteriors
        ]
        super().__init__(components)

    @staticmethod
    def sufficient_statistics(X):
        return NormalDiagonalCovariance.sufficient_statistics(X)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            mean, var)

    def forward(self, T, labels=None):
        feadim = .25 * T.size(1)
        retval = T @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval


class NormalFullCovarianceSet(NormalSet):
    'Set Normal density models with full covariance.'

    def __init__(self, prior, posteriors):
        components = [
            NormalFullCovariance(prior, post) for post in posteriors
        ]
        super().__init__(components)

    @staticmethod
    def sufficient_statistics(X):
        return NormalFullCovariance.sufficient_statistics(X)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalFullCovariance.sufficient_statistics_from_mean_var(mean,
            var)

    def forward(self, T, labels=None):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - T.size(1))))
        retval = T @ self.expected_natural_params_as_matrix().t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval


def normal_diag_natural_params(mean, var):
    '''Transform the standard parameters of a Normal (diag. cov.) into
    their canonical forms.

    Note:
        The (negative) log normalizer is appended to it.

    '''
    return torch.cat([
        -1. / (2 * var),
        mean / var,
        -(mean ** 2) / (2 * var),
        -.5 * torch.log(var)
    ], dim=-1)


class FixedIsotropicGaussian:
    def __init__(self, dim):
        mean = torch.ones(dim)
        var = torch.ones(dim)
        self._np = normal_diag_natural_params(mean, var)

    def expected_natural_params(self, mean, var):
        return self._np, None


#######################################################################
# NormalSet model with shared covariance.
#######################################################################

class NormalSetSharedCovariance(BayesianModel, metaclass=abc.ABCMeta):
    '''Set of Normal density models with a globale shared covariance
    matrix.

    '''

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(X):
        NotImplemented

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics_from_mean_var(mean, var):
        NotImplemented

    def __init__(self, prior, posterior, ncomp):
        super().__init__()
        self._ncomp = ncomp
        self.means_prec_param = BayesianParameter(prior, posterior)

    def __len__(self):
        return self._ncomp


class NormalSetSharedDiagonalCovariance(NormalSetSharedCovariance):
    '''Set of Normal density models with a globale shared full
    covariance matrix.

    '''

    @staticmethod
    def sufficient_statistics(X):
        T1 = torch.cat([X**2, torch.ones_like(X)], dim=1)
        T2 = torch.cat([X, torch.ones_like(X)], dim=1)
        return T1, T2

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        T1 = torch.cat([mean ** 2 + var, torch.ones_like(mean)], dim=1)
        T2 = torch.cat([mean, torch.ones_like(mean)], dim=1)
        return T1, T2

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('expected integer key')
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, D = _jointnormalgamma_split_nparams(
            exp_param, self._ncomp)
        cov = 1 / (-2 * param1)
        mean = cov * param2[key]
        return NormalSetElement(mean=mean, cov=torch.diag(cov))

    def _expected_nparams(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, D = _jointnormalgamma_split_nparams(
            exp_param, self._ncomp)
        return torch.cat([param1.view(-1), param4.view(-1)]), \
            torch.cat([param2, param3], dim=1)

    def expected_natural_params_as_matrix(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, D = _jointnormalgamma_split_nparams(
            exp_param, self._ncomp)
        ones = torch.ones_like(param2)
        return torch.cat([ones * param1, param2, param3, ones * param4], dim=1)

    def forward(self, T, labels=None):
        T1, T2 = T
        feadim = T1.size(1) // 2
        params = self._expected_nparams()
        retval = (T1 @ params[0])[:, None] + T2 @ params[1].t()
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, T, weights):
        T1, T2 = T
        feadim = T1.size(1) // 2
        acc_stats = torch.cat([
            T1[:, :feadim].sum(dim=0),
            (weights.t() @ T2[:, :feadim]).view(-1),
            (weights.t() @ T2[:, feadim:]).view(-1),
            len(T1) * torch.ones(feadim).type(T1.type())
        ])
        return {self.means_prec_param: acc_stats}


class NormalSetSharedFullCovariance(NormalSetSharedCovariance):
    '''Set of Normal density models with a globale shared full
    covariance matrix.

    '''

    @staticmethod
    def sufficient_statistics(X):
        T1 = (X[:, :, None] * X[:, None, :]).view(len(X), -1)
        T2 = torch.cat([X, torch.ones(X.size(0), 1).type(X.type())],
            dim=1)
        return T1, T2

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        raise NotImplementedError()

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('expected integer key')
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, D = _jointnormalwishart_split_nparams(
            exp_param, self._ncomp)
        cov = torch.inverse(-2 * param1)
        mean = cov @ param2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def _expected_nparams(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, D = _jointnormalwishart_split_nparams(
            exp_param, self._ncomp)
        return param1.view(-1), \
            torch.cat([param2, param3[:, None]], dim=1), param4

    def expected_natural_params_as_matrix(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, D = _jointnormalwishart_split_nparams(
            exp_param, self._ncomp)
        ones1 = torch.ones(self._ncomp, D**2).type(param1.type())
        ones2 = torch.ones(self._ncomp, 1).type(param1.type())
        return torch.cat([ones1 * param1.view(-1)[None, :],
            param2, param3.view(-1, 1), ones2 * param4], dim=1)

    def forward(self, T, labels=None):
        T1, T2 = T
        feadim = int(math.sqrt(T1.size(1)))
        params = self._expected_nparams()
        retval = (T1 @ params[0])[:, None] + T2 @ params[1].t() + params[2]
        retval -= .5 * feadim * math.log(2 * math.pi)
        return retval

    def accumulate(self, T, weights):
        T1, T2 = T
        feadim = int(math.sqrt(T1.size(1)))
        acc_stats = torch.cat([
            T1.sum(dim=0),
            (weights.t() @ T2[:, :feadim]).view(-1),
            weights.sum(dim=0),
            len(weights) * torch.ones(1).type(weights.type())
        ])
        return {self.means_prec_param: acc_stats}

