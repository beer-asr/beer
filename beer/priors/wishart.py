'''Implementation of the Dirichlet prior.'''

import math
import torch
from .baseprior import ExpFamilyPrior


def _logdet(mats):
    '''Log determinant of a positive definite matrix.'''
    dim = mats.shape[-1]
    vmats = mats.view(-1, dim, dim)
    retval = []
    for mat in vmats:
        if mat.requires_grad:
            mat.register_hook(lambda grad: .5 * (grad + grad.t()))
        retval.append(2 * torch.log(torch.diag(torch.potrf(mat))).sum().view(1))
    return torch.cat(retval).view(-1, 1)


class WishartPrior(ExpFamilyPrior):
    '''Wishart distribution.

    parameters:
        W: DxD scale matrix
        nu: degree of freedom (dof)

    natural parameters:
        eta1 = - 0.5 * inv(W)
        eta2 = 0.5 * (nu - D - 1)

    sufficient statistics (x is a DxD positive definite matrix):
        T_1(x) = x
        T_2(x) = ln |x|

    '''
    __repr_str = '{classname}(scale={shape}, dof={rate})'

    def __init__(self, scale, dof):
        '''
        Args:
            scale (``torch.Tensor[dim,dim]``)): Scale matrix.
            dof (``torch.tensor[1]``): degree of freedom.
        '''
        dim = len(scale)
        if not dof > len(scale) - 1:
            raise ValueError('Degree of freedom should be greater than '
                             'D - 1. dim={dim}, dof={dof}'.format(dim=dim,
                                                                  dof=dof))
        nparams = self.to_natural_parameters(scale, dof)
        super().__init__(nparams)

    def __repr__(self):
        scale, dof = self.to_std_parameters(self.natural_parameters)
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            shape=repr(scale), rate=repr(dof)
        )

    @property
    def strength(self):
        mean, dof = self.to_std_parameters(self.natural_parameters)
        dim = len(mean)
        return dof - dim + 1

    @strength.setter
    def strength(self, value):
        nparams = self.natural_parameters
        dim = int(math.sqrt(len(nparams)) - 1)
        new_dof = value + dim - 1
        nparams[-1] = .5 * (new_dof - dim - 1)
        self.natural_parameters = nparams

    def expected_value(self):
        scale, dof = self.to_std_parameters(self.natural_parameters)
        return dof * scale

    def to_natural_parameters(self, scale, dof):
        dim = len(scale)
        return torch.cat([
            -.5 * scale.inverse().reshape(-1),
            .5 * (dof - dim - 1).view(1),
        ])

    def _to_std_parameters(self, natural_parameters):
        dim = math.sqrt(len(natural_parameters) - 1)
        np1 = natural_parameters[:-1].reshape((dim, dim))
        np2 = natural_parameters[-1]
        scale = torch.inverse(-2 * np1)
        dof = 2 * np2 + dim + 1
        return scale.contiguous().view((dim, dim)), dof

    def _expected_sufficient_statistics(self):
        scale, dof = self.to_std_parameters(self.natural_parameters)
        dtype, device = scale.dtype, scale.device
        dim = len(scale)
        scale_logdet = _logdet(scale)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        sum_digamma = torch.digamma(.5 * (dof + 1 - seq)).sum()
        return torch.cat([
            dof * scale.view(-1),
            (sum_digamma + dim * math.log(2) + scale_logdet).view(1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters

        scale, dof = self.to_std_parameters(natural_parameters)
        dtype, device = scale.dtype, scale.device
        dim = len(scale)

        lognorm = .5 * dof * _logdet(scale)
        lognorm += .5 * dof * dim * math.log(2)
        lognorm += .25 * dim * (dim - 1) * math.log(math.pi)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        lognorm += torch.lgamma(.5 * (dof + 1 - seq)).sum()

        return lognorm


__all__ = ['WishartPrior']

