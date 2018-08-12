'''Implementation of the Normal-Gamma distribution.'''

import torch
from .baseprior import ExpFamilyPrior


class NormalGammaPrior(ExpFamilyPrior):
    '''Wishart distribution.

    parameters:
        mean: mean (Normal)
        scale: scale of the precision matrix (Normal)
        a: shapes parameter shared across dimension (joint Gamma)
        b: rates parameter for each dimension (joint Gamma)

    natural parameters:
        eta1_i = - 0.5 * scale * mean_i * mean_i - b_i
        eta2 = scale * mean
        eta3 = - 0.5 * scale
        eta4 = a - 0.5

    sufficient statistics (mu, l) (l is the diagonal of the precision):
        T_1(mu, l)_i = l_i
        T_2(mu, l)_i = l_i * mu_i
        T_3(mu, l) = sum(l_i * mu_i * mu_i)
        T_4(mu, l) = sum(ln l_i)

    '''
    __repr_str = '{classname}(mean={mean}, scale={scale}, ' \
                 'shape={shape}, rates={rates})'

    def __init__(self, mean, scale, shape, rates):
        '''
        Args:
            mean (``torch.Tensor[dim]``)): Mean of the Normal.
            scale (``torch.Tensor[1]``): Scaling of the precision
                matrix.
            shape (``torch.Tensor[1]`): Shape parameter of the
                Gamma distribution
            rates (``torch.tensor[dim]``): Rate parameters of the
                Gamma distribution.
        '''
        nparams = self.to_natural_parameters(mean, scale, shape, rates)
        super().__init__(nparams)

    def __repr__(self):
        mean, scale, shape, rates = self.to_std_parameters()
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            mean=repr(mean), scale=repr(scale),
            shape={shape}, rates={rates}
        )

    @property
    def strength(self):
        return 2 * (self.natural_parameters[-1] + .5)

    @strength.setter
    def strength(self, value):
        mean, scale, shape, rates = self.to_std_parameters()
        diag_precision = shape / rates
        scale = torch.tensor(value, dtype=scale.dtype, device=scale.device)
        shape = torch.tensor(.5 * value, dtype=scale.dtype, device=scale.device)
        self.natural_parameters = self.to_natural_parameters(
            mean,
            scale,
            shape,
            shape / diag_precision
        )

    def to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        dim = (len(natural_parameters) - 2) // 2
        np1 = natural_parameters[:dim]
        np2 = natural_parameters[dim:2 * dim]
        np3, np4 = natural_parameters[-2], natural_parameters[-1]

        scale = -2 * np3
        shape = np4 + .5
        mean = np2 / scale
        rates = -np1 - .5 * scale * mean * mean

        return mean, scale, shape, rates

    def to_natural_parameters(self, mean, scale, shape, rates):
        return torch.cat([
            -.5 * scale * mean * mean - rates,
            scale * mean,
            -.5 * scale.view(1),
            shape.view(1) - .5,
        ])

    def expected_sufficient_statistics(self):
        mean, scale, shape, rates = self.to_std_parameters()
        dim = len(mean)
        diag_precision = shape / rates
        logdet = torch.sum(torch.digamma(shape) - torch.log(rates))
        return torch.cat([
            diag_precision,
            diag_precision * mean,
            ((dim / scale) + torch.sum((diag_precision * mean) * mean)).view(1),
            logdet.view(1)
        ])

    def expected_value(self):
        mean, _, shape, rates = self.to_std_parameters()
        return mean, shape / rates

    def log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, scale, shape, rates = self.to_std_parameters(natural_parameters)
        dim = len(mean)
        return dim * torch.lgamma(shape) - shape * rates.log().sum() \
            - .5 * dim * scale.log()


__all__ = ['NormalGammaPrior']
