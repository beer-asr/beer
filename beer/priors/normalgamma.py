'''Implementation of the Normal-Gamma distribution.'''

import torch
from .baseprior import ExpFamilyPrior


class NormalGammaPrior(ExpFamilyPrior):
    '''Normal-Gamma distribution.

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

    def expected_value(self):
        mean, _, shape, rates = self.to_std_parameters()
        return mean.view(-1), shape.view(-1) / rates.view(-1)

    def to_natural_parameters(self, mean, scale, shape, rates):
        return torch.cat([
            -.5 * scale * mean.pow(2) - rates,
            scale * mean,
            -.5 * scale.view(1),
            shape.view(1) - .5,
        ])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        np_dim = natural_parameters.shape[-1]
        natural_parameters = natural_parameters.view(-1, np_dim)

        dim = (natural_parameters.shape[-1] - 2) // 2
        np1 = natural_parameters[:, :dim]
        np2 = natural_parameters[:, dim:2 * dim]
        np3, np4 = natural_parameters[:, -2][:, None], \
                   natural_parameters[:, -1][:, None]

        scale = -2 * np3
        shape = np4 + .5
        mean = np2 / scale
        rates = -np1 - .5 * scale * mean.pow(2)

        return mean, scale, shape, rates

    def _expected_sufficient_statistics(self):
        mean, scale, shape, rates = self.to_std_parameters()
        mean, scale, shape, rates = mean.view(-1), scale.view(-1), \
                                    shape.view(-1), rates.view(-1)
        dim = len(mean)
        diag_precision = shape / rates
        logdet = torch.sum(torch.digamma(shape) - torch.log(rates))
        return torch.cat([
            diag_precision,
            diag_precision * mean,
            ((dim / scale) + (diag_precision * mean.pow(2)).sum()).view(1),
            logdet.view(1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, scale, shape, rates = self.to_std_parameters(natural_parameters)
        dim = mean.shape[-1]
        return dim * torch.lgamma(shape) - shape * rates.log().sum(dim=-1)[:, None] \
            - .5 * dim * scale.log()


class JointNormalGammaPrior(ExpFamilyPrior):
    '''Joint NormalGamma distribution.

    parameters:
        means: mean (Normals)
        scales: scale of the precision matrix (Normals)
        a: shape parameter shared across dimension (joint Gamma)
        b: rates parameter for each dimension (joint Gammas)

    natural parameters:
        eta1_i_k = - 0.5 * scales[k] * means[k]_i * means[k]_i - b_i
        eta2_k = scales[k] * means[k]
        eta3_k = - 0.5 * scales[k]
        eta4 = a - 0.5

    sufficient statistics (mu, l) (l is the diagonal of the precision):
        T_1(mu, l)_i = l_i
        T_2(mu, l)_i_k = l_i * mu[k]_i
        T_3(mu, l)_k = sum(l_i * mu[k]_i * mu[k]_i)
        T_4(mu, l) = sum(ln l_i)

    '''
    __repr_str = '{classname}(means={means}, scale={scale}, ' \
                 'shape={shape}, rates={rates})'

    def __init__(self, means, scales, shape, rates):
        '''
        Args:
            means (``torch.Tensor[dim]``)): Means of the Normals.
            scales (``torch.Tensor[1]``): Scaling of the precision
                matrix.
            shape (``torch.Tensor[1]`): Shape parameter of the
                Gamma distribution
            rates (``torch.tensor[dim]``): Rate parameters of the
                Gamma distribution.
        '''
        self._ncomp, self._dim = means.shape
        nparams = self.to_natural_parameters(means, scales, shape, rates)
        super().__init__(nparams)

    def __repr__(self):
        means, scales, shape, rates = self.to_std_parameters()
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            means=repr(means), scales=repr(scales),
            shape={shape}, rates={rates}
        )

    def expected_value(self):
        means, _, shape, rates = self.to_std_parameters()
        return means, shape / rates

    def to_natural_parameters(self, means, scales, shape, rates):
        return torch.cat([
            -.5 * ((scales[:, None] * means) * means).sum(dim=0) - rates,
            (scales[:, None] * means).view(-1),
            -.5 * scales.view(-1),
            shape.view(1) - 1. + .5 * self._ncomp
        ])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        ncomp, dim = self._ncomp, self._dim
        np1 = natural_parameters[:dim]
        np2s = natural_parameters[dim:dim + ncomp * dim].view((ncomp, dim))
        np3s = natural_parameters[-(ncomp + 1):-1]
        np4 = natural_parameters[-1]

        scales = -2 * np3s
        shape = np4 + 1 - .5 * ncomp
        means = np2s / scales[:, None]
        rates = -np1 - .5 * ((scales[:, None] * means) * means).sum(dim=0)

        return means, scales, shape, rates

    def _expected_sufficient_statistics(self):
        means, scales, shape, rates = self.to_std_parameters()
        dim = self._dim
        diag_precision = shape / rates
        logdet = torch.sum(torch.digamma(shape) - torch.log(rates))
        return torch.cat([
            diag_precision,
            (diag_precision[None] * means).view(-1),
            ((dim / scales) + torch.sum((diag_precision[None] * means) * means,
                                         dim=-1)).view(-1),
            logdet.view(1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        _, scales, shape, rates = self.to_std_parameters(natural_parameters)
        dim = self._dim
        return dim * torch.lgamma(shape) - shape * rates.log().sum() \
            - .5 * dim * scales.log().sum()


__all__ = ['NormalGammaPrior', 'JointNormalGammaPrior']
