'''Implementation of the matrix Normal prior.'''

import math
import torch
from .baseprior import ExpFamilyPrior
from .wishart import _logdet


class MatrixNormalPrior(ExpFamilyPrior):
    '''Matrix Normal prior.

    parameters:
        M: mean of the distribution (n x p matrix)
        U: covariance matrix (n x n positive definite matrix)

    natural parameters:
        eta1 = vec(- 0.5 * inv(U))
        eta2 = vec(inv(U) * M)

    sufficient statistics (W is a nxp real matrix):
        T_1(W) =  vec(W * W^T)
        T_2(W) = vec(W)

    '''
    __repr_str = '{classname}(mean={mean}, cov={cov})'

    def __init__(self, mean, cov):
        '''
        Args:
            mean (``torch.Tensor[dim,dim]``)): Matrix mean.
            cov (``torch.tensor[1]``): Covariance matrix.
        '''
        self.dims = mean.shape
        nparams = self.to_natural_parameters(mean, cov)
        super().__init__(nparams)

    def __repr__(self):
        mean, cov = self.to_std_parameters(self.natural_parameters)
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            mean=repr(mean),
            cov=repr(cov)
        )

    def expected_value(self):
        mean, _ = self.to_std_parameters(self.natural_parameters)
        return mean

    def to_natural_parameters(self, mean, cov):
        prec = cov.inverse().contiguous()
        return torch.cat([-.5 * prec.view(-1),
                          (prec @ mean).contiguous().view(-1)])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        dim1, dim2 = self.dims
        precision = - 2 *natural_parameters[:int(dim1**2)].view(dim1, dim1)
        cov = precision.inverse()
        mean = cov @ natural_parameters[int(dim1**2):].view(dim1, dim2)
        return mean, cov

    def _expected_sufficient_statistics(self):
        mean, cov = self.to_std_parameters(self.natural_parameters)
        return torch.cat([
            (self.dims[1] * cov + mean @ mean.t()).view(-1),
            mean.view(-1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, cov = self.to_std_parameters(natural_parameters)
        precision = cov.inverse()
        log_norm = - self.dims[1] * .5 * _logdet(precision)
        log_norm += .5 * torch.trace(mean.t() @ precision @  mean)
        return log_norm


__all__ = ['MatrixNormalPrior']
