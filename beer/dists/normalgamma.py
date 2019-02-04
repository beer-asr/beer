import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily


__all__ = ['NormalGamma', 'NormalGammaStdParams']


@dataclass(init=False, eq=False, unsafe_hash=True)
class NormalGammaStdParams(torch.nn.Module):
    '''Standard parameterization of the Normal-Gamma pdf.

    Note:
        We use the shape-rate parameterization.

    '''

    mean: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor
    rates: torch.Tensor

    def __init__(self, mean, scale, shape, rates):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('shape', shape)
        self.register_buffer('rates', rates)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        dim = (len(natural_params)- 2) // 2
        np1 = natural_params[:dim]
        np2 = natural_params[dim:2*dim]
        np3 = natural_params[-2]
        np4 = natural_params[-1]
        scale = -2 * np3
        shape = np4 + .5
        mean = np1 / scale
        rates = -np2 - .5 * scale.view(1) * mean**2
        return cls(mean, scale, shape, rates)


class NormalGamma(ExponentialFamily):
    '''Set of independent Normal-Gamma distribution having the same
    scale (Normal) and shape (Gamma) parameters for all dimension.

    '''

    _std_params_def = {
        'mean': 'Mean of the Normal.',
        'scale': 'Scale of the (diagonal) covariance matrix.',
        'shape': 'Shape parameter of the Gamma (shared across dimension).',
        'rates': 'Rate parameters of the Gamma.'
    }

    @property
    def dim(self):
        '''Return a tuple with the dimension of the noram and the
        dimension of the joint Gamma densities.

        '''
        return (len(self.mean), len(self.mean))

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable mu (vector), l (vector with positive
        elements) the sufficient statistics of the Normal-Wishart are
        given by:

        stats = (
            l * mu,
            l,
            \sum_i (l * mu^2)_i,
            \sum_i ln l_i
        )

        For the standard parameters (m=mean, k=scale, a=shape, b=rates)
        expectation of the sufficient statistics is given by:

        E[stats] = (
            (a / b) * m,
            (a / b),
            (D/k) + \sum_i ((a / b) * m^2)_i,
            \sum_i psi(a) - ln(b_i)
        )

        Note: ""D" is the dimenion of "m"
            and "psi" is the "digamma" function.

        '''
        diag_precision = self.shape / self.rates
        logdet = torch.sum(torch.digamma(self.shape) - torch.log(self.rates))
        return torch.cat([
            diag_precision * self.mean,
            diag_precision,
            ((self.dim[0] / self.scale) + \
                (diag_precision * self.mean**2).sum()).reshape(1),
            logdet.reshape(1)
        ])

    def expected_value(self):
        'Expected mean and expected (diagonal) precision matrix.'
        return self.mean, self.shape / self.rates

    def log_norm(self):
        dim = self.dim[0]
        return dim * torch.lgamma(self.shape) \
            - self.shape * self.rates.log().sum(dim=-1) \
            - .5 * dim * self.scale.log()

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (m=mean, k=scale, a=shape, b=rates) the
        natural parameterization is given by:

        nparams = (
            k * m ,
            -.5 * k * m^2
            -.5 * k,
            a - .5
        )

        Note:
            "D" is the dimension of "m" and "^2" is the elementwise
            square operation.

        Returns:
            ``torch.Tensor[2 * D + 2]``

        '''
        return torch.cat([
            self.scale * self.mean,
            -.5 * self.scale * self.mean**2 - self.rates,
            -.5 * self.scale.reshape(1),
            self.shape.reshape(1) - .5,
        ])

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)


class JointNormalGammaPrior(ExponentialFamily):
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

    _std_params_def = {}

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

