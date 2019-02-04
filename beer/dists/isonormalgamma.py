import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily


__all__ = ['IsotropicNormalGamma', 'IsotropicNormalGammaStdParams']


@dataclass(init=False, eq=False, unsafe_hash=True)
class IsotropicNormalGammaStdParams(torch.nn.Module):
    'Standard parameterization of the Normal-Gamma pdf.'

    mean: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor
    rate: torch.Tensor

    def __init__(self, mean, scale, shape, rate):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('shape', shape)
        self.register_buffer('rate', rate)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        dim = len(natural_params) - 3
        np1 = natural_params[:dim]
        np2 = natural_params[dim:dim+1]
        np3 = natural_params[-2].view(1)
        np4 = natural_params[-1].view(1)
        scale = -2 * np3
        shape = np4 + 1 - .5 * dim
        mean = np1 / scale
        rate = -np2 - .5 * scale * torch.sum(mean * mean, dim=-1)
        return cls(mean, scale, shape, rate)


class IsotropicNormalGamma(ExponentialFamily):
    '''Set of independent Normal-Gamma distribution having the same
    scale (Normal), shape and rate (Gamma) parameters for all dimension.

    '''

    _std_params_def = {
        'mean': 'Mean of the Normal.',
        'scale': 'Scale of the (diagonal) covariance matrix.',
        'shape': 'Shape parameter of the Gamma (shared across dimension).',
        'rate': 'Rate parameter of the Gamma (shared across dimension).'
    }

    @property
    def dim(self):
        '''Return a tuple with the dimension of the Normal and the
        dimension of the joint Gamma (just one for the latter).

        '''
        return (len(self.mean), 1)

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable mu (vector), l (positive scalar)
        the sufficient statistics of the isotropic Normal-Gamma are
        given by:

        stats = (
            l * mu,
            l,
            l * \sum_i mu^2_i,
            D * ln(l)
        )

        For the standard parameters (m=mean, k=scale, a=shape, b=rate)
        expectation of the sufficient statistics is given by:

        E[stats] = (
            (a / b) * m,
            (a / b),
            (D/k) + (a / b) * \sum_i m^2_i,
            (psi(a) - ln(b))
        )

        Note: ""D" is the dimenion of "m"
            and "psi" is the "digamma" function.

        '''
        dim = self.dim[0]
        precision = self.shape / self.rate
        logdet = (torch.digamma(self.shape) - torch.log(self.rate))
        return torch.cat([
            precision * self.mean,
            precision.view(1),
            ((dim / self.scale) + precision * (self.mean**2).sum()).view(1),
            logdet.view(1)
        ])

    def expected_value(self):
        'Expected mean and expected precision.'
        return self.mean, self.shape / self.rate

    def log_norm(self):
        dim = self.dim[0]
        return torch.lgamma(self.shape) \
            - self.shape * self.rate.log() \
            - .5 * dim * self.scale.log()

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (m=mean, k=scale, a=shape, b=rate) the
        natural parameterization is given by:

        nparams = (
            k * m ,
            -.5 * k * m^2
            -.5 * k,
            a - 1 + .5 * D
        )

        Note:
            "D" is the dimension of "m" and "^2" is the elementwise
            square operation.

        Returns:
            ``torch.Tensor[D + 3]``

        '''
        return torch.cat([
            self.scale * self.mean,
            (-.5 * self.scale * torch.sum(self.mean**2) - self.rate).view(1),
            -.5 * self.scale.view(1),
            self.shape.view(1) - 1 + .5 * self.dim[0]
        ])

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)


class JointIsotropicNormalGammaPrior:
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

