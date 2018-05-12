
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
    def sufficient_statistics(data):
        '''Compute the sufficient statistics of the data.

        Args:
            data (Tensor): Data.

        Returns:
            (Tensor): Sufficient statistics of the data.

        '''
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    # pylint: disable=c0103
    # Invalid method name.
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
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mean(self):
        'Expected value of the mean w.r.t. posterior distribution.'
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cov(self):
        '''Expected value of the covariance matrix w.r.t posterior
         distribution.

        '''
        raise NotImplementedError


class NormalDiagonalCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([data ** 2, data, torch.ones_like(data),
                          torch.ones_like(data)], dim=-1)

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
        np1, _, _, _ = evalue.view(4, -1)
        return torch.diag(1/(-2 * np1))

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

    def forward(self, s_stats, labels=None):
        feadim = .25 * s_stats.size(1)
        exp_llh = s_stats @ self.mean_prec_param.expected_value
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}


class NormalFullCovariance(Normal):
    'Bayesian Normal distribution with diagonal covariance matrix.'

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([
            (data[:, :, None] * data[:, None, :]).view(len(data), -1),
            data, torch.ones(data.size(0), 1).type(data.type()),
            torch.ones(data.size(0), 1).type(data.type())
        ], dim=-1)

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


class NormalSet(BayesianModel, metaclass=abc.ABCMeta):
    'Set Normal density models.'

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(data):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    # pylint: disable=C0103
    def sufficient_statistics_from_mean_var(mean, var):
        raise NotImplementedError

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

    # pylint: disable=C0103
    # Invalid method name.
    def expected_natural_params_as_matrix(self):
        return torch.cat([param.expected_value[None]
                          for param in self._parameters], dim=0)

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        weights = parent_msg
        return dict(zip(self.parameters, weights.t() @ s_stats))


class NormalDiagonalCovarianceSet(NormalSet):
    'Set Normal density models with diagonal covariance.'

    def __init__(self, prior, posteriors):
        components = [
            NormalDiagonalCovariance(prior, post) for post in posteriors
        ]
        super().__init__(components)

    @staticmethod
    def sufficient_statistics(data):
        return NormalDiagonalCovariance.sufficient_statistics(data)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            mean, var)

    def forward(self, s_stats, labels=None):
        feadim = .25 * s_stats.size(1)
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
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
    def sufficient_statistics(data):
        return NormalFullCovariance.sufficient_statistics(data)

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return NormalFullCovariance.sufficient_statistics_from_mean_var(mean,
                                                                        var)

    def forward(self, s_stats, labels=None):
        feadim = .5 * (-1 + math.sqrt(1 - 4 * (2 - s_stats.size(1))))
        retval = s_stats @ self.expected_natural_params_as_matrix().t()
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


#######################################################################
# NormalSet model with shared covariance.
#######################################################################

class NormalSetSharedCovariance(BayesianModel, metaclass=abc.ABCMeta):
    '''Set of Normal density models with a globale shared covariance
    matrix.

    '''

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(data):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    # pylint: disable=C0103
    def sufficient_statistics_from_mean_var(mean, var):
        raise NotImplementedError

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
    def sufficient_statistics(data):
        s_stats1 = torch.cat([data**2, torch.ones_like(data)], dim=1)
        s_stats2 = torch.cat([data, torch.ones_like(data)], dim=1)
        return s_stats1, s_stats2

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        s_stats1 = torch.cat([mean ** 2 + var, torch.ones_like(mean)], dim=1)
        s_stats2 = torch.cat([mean, torch.ones_like(mean)], dim=1)
        return s_stats1, s_stats2

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('expected integer key')
        exp_param = self.means_prec_param.expected_value
        param1, param2, _, _, _ = _jointnormalgamma_split_nparams(
            exp_param, self._ncomp)
        cov = 1 / (-2 * param1)
        mean = cov * param2[key]
        return NormalSetElement(mean=mean, cov=torch.diag(cov))

    def _expected_nparams(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, _ = _jointnormalgamma_split_nparams(
            exp_param, self._ncomp)
        return torch.cat([param1.view(-1), param4.view(-1)]), \
            torch.cat([param2, param3], dim=1)

    # pylint: disable=C0103
    # Invalid method name.
    def expected_natural_params_as_matrix(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, _ = _jointnormalgamma_split_nparams(
            exp_param, self._ncomp)
        ones = torch.ones_like(param2)
        return torch.cat([ones * param1, param2, param3, ones * param4], dim=1)

    def forward(self, s_stats, labels=None):
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


class NormalSetSharedFullCovariance(NormalSetSharedCovariance):
    '''Set of Normal density models with a globale shared full
    covariance matrix.

    '''

    @staticmethod
    def sufficient_statistics(data):
        s_stats1 = (data[:, :, None] * data[:, None, :]).view(len(data), -1)
        s_stats2 = torch.cat([data, torch.ones(data.size(0), 1).type(data.type())],
                             dim=1)
        return s_stats1, s_stats2

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        raise NotImplementedError()

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('expected integer key')
        exp_param = self.means_prec_param.expected_value
        param1, param2, _, _, _ = _jointnormalwishart_split_nparams(
            exp_param, self._ncomp)
        cov = torch.inverse(-2 * param1)
        mean = cov @ param2[key]
        return NormalSetElement(mean=mean, cov=cov)

    def _expected_nparams(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, _ = _jointnormalwishart_split_nparams(
            exp_param, self._ncomp)
        return param1.view(-1), \
            torch.cat([param2, param3[:, None]], dim=1), param4

    # pylint: disable=C0103
    def expected_natural_params_as_matrix(self):
        exp_param = self.means_prec_param.expected_value
        param1, param2, param3, param4, D = _jointnormalwishart_split_nparams(
            exp_param, self._ncomp)
        ones1 = torch.ones(self._ncomp, D**2).type(param1.type())
        ones2 = torch.ones(self._ncomp, 1).type(param1.type())
        return torch.cat([ones1 * param1.view(-1)[None, :],
                          param2, param3.view(-1, 1), ones2 * param4], dim=1)

    def forward(self, s_stats, labels=None):
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
