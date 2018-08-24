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

    def expected_value(self):
        mean, _, mean_precision, dof = self.to_std_parameters()
        return mean, dof * mean_precision

    def to_natural_parameters(self, mean, scale, mean_precision, dof):
        dim = len(mean)
        inv_mean_prec = mean_precision.inverse()
        return torch.cat([
            -.5 * (scale * torch.ger(mean, mean) + inv_mean_prec).reshape(-1),
            scale * mean,
            -.5 * scale.view(1),
            .5 * (dof - dim).view(1),
        ])

    def _to_std_parameters(self, natural_parameters=None):
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

    def _expected_sufficient_statistics(self):
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

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters

        mean, scale, mean_precision, dof = \
            self.to_std_parameters(natural_parameters)
        dtype, device = mean.dtype, mean.device
        dim = len(mean)

        lognorm =  .5 * dof * _logdet(mean_precision)
        lognorm -= .5 * dim * torch.log(scale)
        lognorm += .5 * dof * dim * math.log(2)
        lognorm += .25 * dim * (dim - 1) * math.log(math.pi)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device,
                           requires_grad=False)
        lognorm += torch.lgamma(.5 * (dof + 1 - seq)).sum()
        return lognorm


class JointNormalWishartPrior(ExpFamilyPrior):
    '''Wishart distribution.

    parameters:
        means mean for each normal (Normals)
        scales: scale of the precision matrix for each norm (Normals)
        W: DxD "mean" of the precision matrix, i.e. the
            scale matrix (Wishart)
        nu: degree of freedom (Wishart)

    natural parameters:
        eta1 = - 0.5 * (inv(W) - sum(scales[k] * means[k] * means[k]^T)
        eta2[k] = scales[k] * means[k]
        eta3[k] = - 0.5 * scales[k]
        eta4 = 0.5 * (nu - D - 1 + k)

    sufficient statistics (mu, L):
        T_1(x) = L
        T_2(x) = L * mu
        T_3(x) = trace(L * mu * mu^T)
        T_4(x) = ln |L|

    '''
    __repr_str = '{classname}(means={means}, scales={scales}, ' \
                 'mean_precision={mean_precision}, dof={dof})'

    def __init__(self, means, scales, mean_precision, dof):
        '''
        Args:
            means (``torch.Tensor[dim]``)): Mean for each Normal.
            scales (``torch.Tensor[1]``): Scaling of the precision
                matrix for each Normal.
            mean_precision (``torch.Tensor[dim,dim]`): Mean of the
                precision matrix.
            dof (``torch.tensor[1]``): degree of freedom.
        '''
        self._ncomp, self._dim = means.shape
        if not dof > self._dim - 1:
            raise ValueError('Degree of freedom should be greater than '
                             'D - 1. dim={dim}, dof={dof}'.format(dim=self._dim,
                                                                  dof=dof))
        nparams = self.to_natural_parameters(means, scales, mean_precision, dof)
        super().__init__(nparams)

    def __repr__(self):
        means, scales, mean_precision, dof = \
            self.to_std_parameters(self.natural_parameters)
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            means=repr(means), scales=repr(scales),
            mean_precision={mean_precision}, dof={dof}
        )

    def expected_value(self):
        means, _, mean_precision, dof = self.to_std_parameters()
        return means, dof * mean_precision

    def to_natural_parameters(self, means, scales, mean_precision, dof):
        ncomp, dim = self._ncomp, self._dim
        inv_mean_prec = mean_precision.inverse()
        quad_means = (scales[:, None] * means).t() @ means
        return torch.cat([
            -.5 * (quad_means + inv_mean_prec).view(-1),
            (scales[:, None] * means).view(-1),
            -.5 * scales.view(-1),
            .5 * (dof - dim - 1 + ncomp).view(1),
        ])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        ncomp, dim = self._ncomp, self._dim
        np1 = natural_parameters[:int(dim**2)].view((dim, dim))
        np2s = natural_parameters[int(dim**2):int(dim**2) + ncomp * dim].view((ncomp, dim))
        np3s = natural_parameters[-(ncomp+1):-1]
        np4 = natural_parameters[-1]

        scales = -2 * np3s
        dof = 2 * np4 + dim + 1 - ncomp
        means = np2s / scales[:, None]
        quad_means = (scales[:, None] * means).t() @ means
        mean_precision = torch.inverse(-2 * np1 - quad_means)

        return means, scales, mean_precision, dof

    def _expected_sufficient_statistics(self):
        means, scales, mean_precision, dof = self.to_std_parameters()
        dtype, device = means.dtype, means.device
        ncomp, dim = self._ncomp, self._dim

        precision = dof * mean_precision
        logdet = _logdet(mean_precision)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        sum_digamma = torch.digamma(.5 * (dof + 1 - seq)).sum()

        quad_means = (means[:, :, None] @ means[:, None, :]).view(ncomp, -1)
        vec_precision = precision.view(-1)
        return torch.cat([
            precision.reshape(-1),
            (means @ precision).view(-1),
            (dim / scales) + quad_means @ vec_precision,
            (sum_digamma + dim * math.log(2) + logdet).view(1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters

        _, scales, mean_precision, dof = \
            self.to_std_parameters(natural_parameters)
        dtype, device = mean_precision.dtype, mean_precision.device
        dim = self._dim

        lognorm = .5 * dof * _logdet(mean_precision)
        lognorm -= .5 * dim  * torch.log(scales).sum()
        lognorm += .5 * dof * dim * math.log(2)
        lognorm += .25 * dim * (dim - 1) * math.log(math.pi)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        lognorm += torch.lgamma(.5 * (dof + 1 - seq)).sum()
        return lognorm


__all__ = ['NormalWishartPrior', 'JointNormalWishartPrior']

