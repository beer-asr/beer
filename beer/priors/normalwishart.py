'''Implementation of the Normal-Wishart distribution.'''

import math
import torch
from .baseprior import ExpFamilyPrior
from .wishart import _logdet


class NormalWishartPrior(ExpFamilyPrior):
    '''Wishart distribution.

    parameters:
        mean: mean (Normal)
        scale: scale of the precision matrix (Normal)
        W: DxD "mean" of the precision matrix, i.e. the
            scale matrix (Wishart)
        nu: degree of freedom (Wishart)

    natural parameters:
        eta1 = - 0.5 * (inv(W) - scale * mean * mean^T)
        eta2 = scale * mean
        eta3 = - 0.5 * scale
        eta4 = 0.5 * (nu - D)

    sufficient statistics (mu, L):
        T_1(x) = L
        T_2(x) = L * mu
        T_3(x) = trace(L * mu * mu^T)

    '''
    __repr_str = '{classname}(mean={mean}, scale={scale}, ' \
                 'mean_precision={mean_precision}, dof={dof})'

    def __init__(self, mean, scale, mean_precision, dof):
        '''
        Args:
            mean (``torch.Tensor[dim]``)): Mean of the Normal.
            scale (``torch.Tensor[1]``): Scaling of the precision
                matrix.
            mean_precision (``torch.Tensor[dim,dim]`): Mean of the
                precision matrix.
            dof (``torch.tensor[1]``): degree of freedom.
        '''
        if not dof > len(mean) - 1:
            raise ValueError('Degree of freedom should be greater than '
                             'D - 1. dim={dim}, dof={dof}'.format(dim=dim,
                                                                  dof=dof))
        nparams = self.to_natural_parameters(mean, scale, mean_precision, dof)
        super().__init__(nparams)

    def __repr__(self):
        mean, scale, mean_precision, dof = \
            self.to_std_parameters(self.natural_parameters)
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            mean=repr(mean), scale=repr(scale),
            mean_precision={mean_precision}, dof={dof}
        )

    @property
    def strength(self):
        return 2 * self.natural_parameters[-1] + 1

    @strength.setter
    def strength(self, value):
        mean, scale, mean_precision, dof = self.to_std_parameters()
        mean_precision *= dof
        scale = torch.tensor(value, dtype=scale.dtype, device=scale.device)
        dof = torch.tensor(value + len(mean) - 1, dtype=scale.dtype,
                           device=scale.device)
        self.natural_parameters = self.to_natural_parameters(
            mean,
            scale,
            mean_precision / dof,
            dof
        )

    def to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        dim = int(-1 + math.sqrt(1 - 4 * (2 - len(natural_parameters)))) // 2
        np1 = natural_parameters[:int(dim**2)].reshape((dim, dim))
        np2 = natural_parameters[int(dim**2):int(dim**2) + dim]
        np3, np4 = natural_parameters[-2], natural_parameters[-1]

        scale = -2 * np3
        dof = 2 * np4 + dim
        mean = np2 / scale
        mean_precision = torch.inverse(-2 * np1 - scale * torch.ger(mean, mean))

        return mean, scale, mean_precision, dof

    def to_natural_parameters(self, mean, scale, mean_precision, dof):
        dim = len(mean)
        inv_mean_prec = mean_precision.inverse()
        return torch.cat([
            -.5 * (scale * torch.ger(mean, mean) + inv_mean_prec).reshape(-1),
            scale * mean,
            -.5 * scale.view(1),
            .5 * (dof - dim).view(1),
        ])

    def expected_sufficient_statistics(self):
        mean, scale, mean_precision, dof = self.to_std_parameters()
        dtype, device = mean.dtype, mean.device
        dim = len(mean)

        precision = dof * mean_precision
        logdet = _logdet(mean_precision)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        sum_digamma = torch.digamma(.5 * (dof + 1 - seq)).sum()

        return torch.cat([
            precision.reshape(-1),
            precision @ mean,
            (dim / scale) + torch.trace(precision @ torch.ger(mean, mean)).view(1),
            (sum_digamma + dim * math.log(2) + logdet).view(1)
        ])

    def expected_value(self):
        mean, _, mean_precision, dof = self.to_std_parameters()
        return mean, dof * mean_precision

    def log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters

        mean, _, mean_precision, dof = \
            self.to_std_parameters(natural_parameters)
        dtype, device = mean.dtype, mean.device
        dim = len(mean)

        lognorm = .5 * dof * _logdet(mean_precision)
        lognorm += .5 * dof * dim * math.log(2)
        lognorm += .25 * dim * (dim - 1) * math.log(math.pi)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        lognorm += torch.lgamma(.5 * (dof + 1 - seq)).sum()
        return lognorm


__all__ = ['NormalWishartPrior']
