'''Implementation of the Normaldistribution.'''

import math
import torch
from .baseprior import ExpFamilyPrior
from .wishart import _logdet


class NormalFullCovariancePrior(ExpFamilyPrior):
    '''Normal distribution with full covariance matrix.

    parameters:
        mean: mean of the distribution
        cov: covariatnce matrix

    natural parameters:
        eta1 = cov^{-1} * mean
        eta2 = - 0.5 * cov^{-1}

    sufficient statistics:
        T_1(x) = x
        T_2(x) = x * x^T

    '''

    def __init__(self, mean, cov):
        self._dim = len(mean)
        nparams = self.to_natural_parameters(mean, cov)
        super().__init__(nparams)

    def __repr__(self):
        return f'{self.__class__.__qualname__}(mean={self.mean}, cov={self.cov})'

    @property
    def dim(self):
        return self._dim

    @property
    def mean(self):
        return self.to_std_parameters(self.natural_parameters)[0]

    @property
    def cov(self):
        return self.to_std_parameters(self.natural_parameters)[1]

    def moments(self):
        stats = self.expected_sufficient_statistics()
        return stats[:self.dim], stats[self.dim:]

    def expected_value(self):
        mean, _ = self.to_std_parameters(self.natural_parameters)
        return mean

    def to_natural_parameters(self, mean, cov):
        prec = cov.inverse()
        return torch.cat([prec @ mean, -.5 * prec.reshape(-1)])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        precision = - 2 * natural_parameters[self.dim:]
        cov = precision.reshape(self.dim, self.dim).inverse()
        mean = cov @ natural_parameters[:self.dim:]
        return mean, cov

    def _expected_sufficient_statistics(self):
        mean, cov = self.to_std_parameters(self.natural_parameters)
        return torch.cat([
            mean,
            (cov + torch.ger(mean, mean)).reshape(-1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, cov = self.to_std_parameters(natural_parameters)
        precision = -2 * natural_parameters[self.dim:]
        precision = precision.reshape(self.dim, self.dim)
        log_norm = .5 * mean @ precision @ mean
        log_norm -= .5 * _logdet(precision).sum()
        return log_norm + .5 * self.dim * math.log(2*math.pi)


class NormalDiagonalCovariancePrior(ExpFamilyPrior):
    '''Normal distribution with diagonal covariance matrix.

    parameters:
        mean: mean of the distribution
        dcov: diagonal of the covariance matrix

    natural parameters:
        eta1 = (1/dcov) * mean
        eta2 = - (1/(2 * dcov)

    sufficient statistics:
        T_1(x) = x
        T_2(x) = diag(x * x^T)

    '''

    def __init__(self, mean, dcov):
        self._dim = len(mean)
        nparams = self.to_natural_parameters(mean, dcov)
        super().__init__(nparams)

    def __repr__(self):
        return f'{self.__class__.__qualname__}(mean={self.mean}, dcov={self.dcov})'

    @property
    def dim(self):
        return self._dim

    @property
    def mean(self):
        return self.to_std_parameters(self.natural_parameters)[0]

    @property
    def dcov(self):
        return self.to_std_parameters(self.natural_parameters)[1]

    def moments(self):
        stats = self.expected_sufficient_statistics()
        return stats[:self.dim], stats[self.dim:]

    def expected_value(self):
        mean, _ = self.to_std_parameters(self.natural_parameters)
        return mean

    def to_natural_parameters(self, mean, dcov):
        prec = 1/dcov
        return torch.cat([prec * mean, -.5 * prec])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        precision = - 2 * natural_parameters[self.dim:]
        dcov = 1/precision
        mean = dcov * natural_parameters[:self.dim]
        return mean, dcov

    def _expected_sufficient_statistics(self):
        mean, dcov = self.to_std_parameters(self.natural_parameters)
        return torch.cat([
            mean,
            (dcov + mean**2)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, dcov = self.to_std_parameters(natural_parameters)
        precision = 1. / dcov
        log_norm = .5 * (precision * mean**2).sum()
        log_norm -= .5 * precision.log().sum()
        return log_norm + .5 * self.dim * math.log(2*math.pi)


class NormalIsotropicCovariancePrior(ExpFamilyPrior):
    '''Normal distribution with isotropic covariance matrix.

    parameters:
        mean: mean of the distribution
        var: variance matrix

    natural parameters:
        eta1 = (1/var) * mean
        eta2 = - (1/(2 * var))

    sufficient statistics:
        T_1(x) = x
        T_2(x) = x^T * x

    '''

    def __init__(self, mean, var):
        self._dim = len(mean)
        nparams = self.to_natural_parameters(mean, var)
        super().__init__(nparams)

    def __repr__(self):
        return f'{self.__class__.__qualname__}(mean={self.mean}, var={self.var})'

    @property
    def dim(self):
        return self._dim

    @property
    def mean(self):
        return self.to_std_parameters(self.natural_parameters)[0]

    @property
    def var(self):
        return self.to_std_parameters(self.natural_parameters)[1]

    def moments(self):
        stats = self.expected_sufficient_statistics()
        return stats[:self.dim], stats[self.dim:]

    def expected_value(self):
        mean, _ = self.to_std_parameters(self.natural_parameters)
        return mean

    def to_natural_parameters(self, mean, var):
        prec = 1/var
        return torch.cat([prec * mean, -.5 * prec.reshape(-1)])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        precision = - 2 * natural_parameters[self.dim:]
        var = 1/precision
        mean = var * natural_parameters[:self.dim]
        return mean, var

    def _expected_sufficient_statistics(self):
        mean, var = self.to_std_parameters(self.natural_parameters)
        return torch.cat([
            mean,
            (self.dim * var + (mean**2).sum()).reshape(-1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, cov = self.to_std_parameters(natural_parameters)
        precision = -2 * natural_parameters[self.dim:]
        log_norm = .5 * precision * (mean**2).sum()
        log_norm -= .5 * self.dim * precision.log()
        return log_norm + .5 * self.dim * math.log(2*math.pi)


__all__ = ['NormalFullCovariancePrior', 'NormalDiagonalCovariancePrior',
           'NormalIsotropicCovariancePrior']

