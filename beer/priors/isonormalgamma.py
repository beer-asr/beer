'''Implementation of the isotropic Normal-Gamma distribution.'''

import torch
from .baseprior import ExpFamilyPrior


class IsotropicNormalGammaPrior(ExpFamilyPrior):
    '''Wishart distribution.

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
        dim = len(self.natural_parameters) - 3
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

    def to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        dim = len(natural_parameters) - 3
        np1 = natural_parameters[0]
        np2 = natural_parameters[1:1 + dim]
        np3, np4 = natural_parameters[-2], natural_parameters[-1]
        scale = -2 * np3
        shape = np4 + 1 - .5 * dim
        mean = np2 / scale
        rate = -np1 - .5 * scale * torch.sum(mean * mean)
        return mean, scale, shape, rate

    def to_natural_parameters(self, mean, scale, shape, rate):
        return torch.cat([
            (-.5 * scale * torch.sum(mean * mean) - rate).view(1),
            scale * mean,
            -.5 * scale.view(1),
            shape.view(1) - 1 + .5 * len(mean),
        ])

    def expected_sufficient_statistics(self):
        mean, scale, shape, rate = self.to_std_parameters()
        dim = len(mean)
        precision = shape / rate
        logdet = torch.digamma(shape) - torch.log(rate)
        return torch.cat([
            precision.view(1),
            precision * mean,
            ((dim / scale) + precision * (mean * mean).sum()).view(1),
            logdet.view(1)
        ])

    def expected_value(self):
        mean, _, shape, rate = self.to_std_parameters()
        return mean, shape / rate

    def log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, scale, shape, rate = self.to_std_parameters(natural_parameters)
        dim = len(mean)
        return torch.lgamma(shape) - shape * rate.log()  - .5 * dim * scale.log()


__all__ = ['IsotropicNormalGammaPrior']
