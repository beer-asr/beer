import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily


__all__ = ['NormalGamma', 'NormalGammaStdParams',
           'JointNormalGammaStdParams', 'JointNormalGamma']


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
        '''Return a tuple with the dimension of the Normal and the
        dimension of the joint Gamma densities.

        '''
        return (len(self.mean), len(self.mean))

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable mu (vector), l (vector with positive
        elements) the sufficient statistics of the Normal-Gamma are
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


@dataclass(init=False, eq=False, unsafe_hash=True)
class JointNormalGammaStdParams(torch.nn.Module):
    means: torch.Tensor
    scales: torch.Tensor
    shape: torch.Tensor
    rates: torch.Tensor

    def __init__(self, means, scales, shape, rates):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('scales', scales)
        self.register_buffer('shape', shape)
        self.register_buffer('rates', rates)

    @classmethod
    def from_natural_parameters(cls, natural_params, ncomp):
        dim = (len(natural_params) - ncomp - 1) // (ncomp + 1)
        np1s = natural_params[:ncomp * dim].reshape(ncomp, dim)
        np2 = natural_params[ncomp * dim : (ncomp + 1) * dim]
        np3s = natural_params[-(ncomp + 1):-1]
        np4 = natural_params[-1]
        scales = -2 * np3s
        shape = np4 + 1 - .5 * ncomp
        means = np1s / scales[:, None]
        rates = -np2 - .5 * ((scales[:, None] * means) * means).sum(dim=0)
        return cls(means, scales, shape, rates)


class JointNormalGamma(ExponentialFamily):
    '''Set of Normal distributions sharing the same Gamma prior over
    the diagonal of the precision matrix.

    '''

    _std_params_def = {
        'means': 'Set of mean parameters.',
        'scales': 'Set of scaling of the precision (for each Normal).',
        'shape': 'Shape parameter (Gamma).',
        'rates': 'Rate parameters (Gamma).'
    }

    @property
    def dim(self):
        '''Return a tuple ((K, D), D)' where K is the number of Normal
        and D is the dimension of their support.

        '''
        return (tuple(self.means.shape), self.means.shape[-1])

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variables mu (set of vector), l (vector of
        positive values) the sufficient statistics of the joint
        Normal-Gamma with diagonal precision matrix are given by:

        stats = (
            l_1 * mu_1,
            ...,
            l_k * mu_k
            l,
            l * mu^2_i,
            \sum_i ln(l)_i
        )

        For the standard parameters (m=mean, k=scale, a=shape, b=rate)
        expectation of the sufficient statistics is given by:

        E[stats] = (
            (a / b) * m,
            (a / b),
            (D/k) + (a / b) * \sum_i m^2_i,
            (psi(a) - ln(b))
        )

        Note: ""D" is the dimenion of "m", "k" is the number of Normal,
            and "psi" is the "digamma" function.

.       '''
        dim = self.dim[1]
        diag_precision = self.shape / self.rates
        logdet = torch.sum(torch.digamma(self.shape) - torch.log(self.rates))
        return torch.cat([
            (diag_precision[None] * self.means).reshape(-1),
            diag_precision,
            ((dim / self.scales) + torch.sum(diag_precision[None] * self.means**2,
                                             dim=-1)).reshape(-1),
            logdet.view(1)
        ])

    def expected_value(self):
        'Expected means and expected diagonal of the precision matrix.'
        return self.means, self.shape / self.rates

    def log_norm(self):
        dim = self.dim[1]
        return dim * torch.lgamma(self.shape) \
            - self.shape * self.rates.log().sum() \
            - .5 * dim * self.scales.log().sum()

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        ncomp = self.dim[0][0]
        return torch.cat([
            (self.scales[:, None] * self.means).view(-1),
            -.5 * (self.scales[:, None] * self.means**2).sum(dim=0) \
                - self.rates,
            -.5 * self.scales.view(-1),
            self.shape.view(1) - 1. + .5 * ncomp
        ])

    def update_from_natural_parameters(self, natural_params):
        ncomp = self.dim[0][0]
        self.params = self.params.from_natural_parameters(natural_params, ncomp)

