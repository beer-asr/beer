'''Implementation of the Normaldistribution.'''

import math
import torch
from .baseprior import ExpFamilyPrior


class NormalFullCovariancePrior(ExpFamilyPrior):
    '''Normal distribution with full covariance matrix.

    parameters:
        mean: mean of the distribution
        scale: scale of the precision matrix (scalar)
        precision: Precision matrix (given by another distribution)

    natural parameters:
        eta1 = scale * mean
        eta2 = - 0.5 * scale

    sufficient statistics (x is a DxD positive definite matrix):
        T_1(x) = precision * mean
        T_2(x) = mean^T * precision * mean

    '''
    __repr_str = '{classname}(mean={shape}, scale={rate}, precision_prior={precision})'

    def __init__(self, mean, scale, precision_prior):
        '''
        Args:
            mean (``torch.Tensor[dim,dim]``)): Scale matrix.
            scale (``torch.tensor[1]``): Scale of the precision matrix.
            precision_prior (:any:`Wishart`): Prior over the precision.
        '''
        self.precision_prior = precision_prior
        nparams = self.to_natural_parameters(mean, scale)
        super().__init__(nparams)

    def __repr__(self):
        mean, scale = self.to_std_parameters(self.natural_parameters)
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            shape=repr(mean), rate=repr(scale),
            precision=repr(self.precision_prior)
        )

    def expected_value(self):
        mean, _ = self.to_std_parameters(self.natural_parameters)
        return mean

    def to_natural_parameters(self, mean, scale):
        return torch.cat([scale * mean, -.5 * scale.view(1)])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        scale = - 2 * natural_parameters[-1]
        mean = natural_parameters[:-1] / scale
        return mean, scale

    def _expected_sufficient_statistics(self):
        mean, scale = self.to_std_parameters(self.natural_parameters)
        dim = len(mean)
        precision = self.precision_prior.expected_value()
        mean_quad = torch.trace(precision @ torch.ger(mean, mean))
        return torch.cat([
            precision.inverse() @ mean,
            (mean_quad + dim / scale).view(1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, scale = self.to_std_parameters(natural_parameters)
        dim = len(mean)
        precision_stats = self.precision_prior.expected_sufficient_statistics()
        precision = precision_stats[:-1].reshape((dim, dim))
        logdet_precision = precision_stats[-1]
        log_norm = .5 * scale * torch.trace(torch.ger(mean, mean) @ precision)
        log_norm -= .5 * logdet_precision
        log_norm -= .5 * dim * scale.log()
        return log_norm


__all__ = ['NormalFullCovariancePrior']
