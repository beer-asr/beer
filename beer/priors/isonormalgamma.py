'''Implementation of the isotropic Normal-Gamma distribution.'''

import torch
from .baseprior import ExpFamilyPrior


class IsotropicNormalGammaPrior(ExpFamilyPrior):
    '''Isotropic NormalGamma distribution.

    parameters:
        mean: mean (Normal)
        scale: scale of the precision matrix (Normal)
        a: shape parameter  (Gamma)
        b: rates parameter (Gamma)

    natural parameters:
        eta1 = - 0.5 * scale * mean^T mean - b
        eta2 = scale * mean
        eta3 = - 0.5 * scale
        eta4 = a - 1 + 0.5 * dim

    sufficient statistics (mu, l):
        T_1(mu, l) = l
        T_2(mu, l) = l * mu
        T_3(mu, l) = l * mu^T mu
        T_4(mu, l) = ln l

    '''
    __repr_str = '{classname}(mean={mean}, scale={scale}, ' \
                 'shape={shape}, rate={rate})'

    def __init__(self, mean, scale, shape, rate):
        '''
        Args:
            mean (``torch.Tensor[dim]``)): Mean of the Normal.
            scale (``torch.Tensor[1]``): Scaling of the precision
                matrix.
            shape (``torch.Tensor[1]`): Shape parameter of the Gamma
                distribution.
            rate (``torch.tensor[dim]``): Rate parameter of the Gamma
                distribution.
        '''
        nparams = self.to_natural_parameters(mean, scale, shape, rate)
        super().__init__(nparams)

    def __repr__(self):
        mean, scale, shape, rate = self.to_std_parameters()
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            mean=repr(mean), scale=repr(scale),
            shape={shape}, rate={rate}
        )

    @property
    def strength(self):
        return self.natural_parameters[-1]

    @strength.setter
    def strength(self, value):
        mean, scale, shape, rate = self.to_std_parameters()
        dim = len(mean)
        precision = shape / rate
        scale = torch.tensor(value, dtype=scale.dtype, device=scale.device)
        shape = torch.tensor(.5 * dim * value, dtype=scale.dtype, device=scale.device)
        self.natural_parameters = self.to_natural_parameters(
            mean,
            scale,
            shape,
            shape / precision
        )

    def expected_value(self):
        mean, _, shape, rate = self.to_std_parameters()
        return mean.view(-1), shape.view(-1) / rate.view(-1)

    def to_natural_parameters(self, mean, scale, shape, rate):
        return torch.cat([
            (-.5 * scale * torch.sum(mean * mean) - rate).view(1),
            scale * mean,
            -.5 * scale.view(1),
            shape.view(1) - 1 + .5 * len(mean),
        ])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        np_dim = natural_parameters.shape[-1]
        natural_parameters = natural_parameters.view(-1, np_dim)

        dim = natural_parameters.shape[-1] - 3
        np1 = natural_parameters[:, 0].view(-1, 1)
        np2 = natural_parameters[:, 1:1 + dim]
        np3, np4 = natural_parameters[:, -2].view(-1, 1), \
                   natural_parameters[:, -1].view(-1, 1)
        scale = -2 * np3
        shape = np4 + 1 - .5 * dim
        mean = np2 / scale
        rate = -np1 - .5 * scale * torch.sum(mean * mean, dim=-1)[:, None]
        return mean, scale, shape, rate

    def _expected_sufficient_statistics(self):
        mean, scale, shape, rate = self.to_std_parameters()
        mean, scale, shape, rate = mean[0], scale[0], shape[0], rate[0]
        dim = len(mean)
        precision = shape / rate
        logdet = torch.digamma(shape) - torch.log(rate)
        return torch.cat([
            precision.view(1),
            precision * mean,
            ((dim / scale) + precision * (mean.pow(2)).sum()).view(1),
            logdet.view(1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters

        mean, scale, shape, rate = self.to_std_parameters(natural_parameters)
        dim = mean.shape[-1]
        return torch.lgamma(shape) - shape * rate.log()  - .5 * dim * scale.log()


class JointIsotropicNormalGammaPrior(ExpFamilyPrior):
    '''Joint isotropic NormalGamma  distribution. The set of normal
    distribution shared the same gamma prior over the precision.

    parameters:
        means: Set of means (Normals)
        scales: Set of scales of the precision matrix (Normals)
        a: shape parameter  (Gamma)
        b: rates parameter (Gamma)

    natural parameters:
        eta1 = - 0.5 * sum_k(scales[k] * mean[k]^T mean[k]) - b
        eta2_k = scales[k] * means[k]
        eta3 = - 0.5 * scales[k]
        eta4 = a - 1 + 0.5 * dim * K

    sufficient statistics (mu, l):
        T_1(mu, l) = l
        T_2(mu, l) = l * mu[k]
        T_3(mu, l) = l * mu[k]^T mu[k]
        T_4(mu, l) = ln l

    '''
    __repr_str = '{classname}(means={means}, scales={scales}, ' \
                 'shape={shape}, rate={rate})'

    def __init__(self, means, scales, shape, rate):
        '''
        Args:
            means (``torch.Tensor[k,dim]``)): Mean of the Normals.
            scales (``torch.Tensor[1]``): Scaling of the precision
                matrix for each Normal.
            shape (``torch.Tensor[1]`): Shape parameter of the Gamma
                distribution.
            rate (``torch.tensor[dim]``): Rate parameter of the Gamma
                distribution.
        '''
        self._ncomp = len(means)
        nparams = self.to_natural_parameters(means, scales, shape, rate)
        super().__init__(nparams)

    def __repr__(self):
        means, scales, shape, rate = self.to_std_parameters()
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            means=repr(means), scales=repr(scales),
            shape={shape}, rate={rate}
        )

    def expected_value(self):
        means, _, shape, rate = self.to_std_parameters()
        return means, shape / rate

    def to_natural_parameters(self, means, scales, shape, rate):
        return torch.cat([
            (-.5 * (scales * (means * means).sum(dim=-1)).sum() - rate).view(1),
            (scales[:, None] * means).view(-1),
            -.5 * scales.view(-1),
            shape.view(1) - 1 + .5 * means.shape[1] * self._ncomp,
        ])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        dim = (len(natural_parameters) - 2 - self._ncomp) // self._ncomp
        np1 = natural_parameters[0]
        np2s = natural_parameters[1:1 + self._ncomp * dim].view(self._ncomp, dim)
        np3s = natural_parameters[-(self._ncomp + 1):-1]
        np4 = natural_parameters[-1]
        scales = -2 * np3s
        shape = np4 + 1 - .5 * dim * self._ncomp
        means = np2s / scales[:, None]
        rate = -np1 - .5 * (scales * (means * means).sum(dim=-1)).sum()
        return means, scales, shape, rate

    def _expected_sufficient_statistics(self):
        means, scales, shape, rate = self.to_std_parameters()
        dim = means.shape[1]
        precision = shape / rate
        logdet = torch.digamma(shape) - torch.log(rate)
        return torch.cat([
            precision.view(1),
            precision * means.view(-1),
            ((dim / scales) + precision * (means * means).sum(dim=-1)).view(-1),
            logdet.view(1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        means, scales, shape, rate = self.to_std_parameters(natural_parameters)
        dim = means.shape[1]
        return torch.lgamma(shape) - shape * rate.log() \
            - .5 * dim * scales.log().sum()


__all__ = ['IsotropicNormalGammaPrior', 'JointIsotropicNormalGammaPrior']
