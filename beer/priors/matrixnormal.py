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

    def __init__(self, mean, cov):
        '''
        Args:
            mean (``torch.Tensor[dim,dim]``)): Matrix mean.
            cov (``torch.tensor[1]``): Covariance matrix.
        '''
        self._dim = mean.shape
        nparams = self.to_natural_parameters(mean, cov)
        super().__init__(nparams)

    def __repr__(self):
        return f'{self.__class__.__qualname__}(mean={self.mean}, cov={self.cov})'

    @property
    def dim(self):
        return self._dim

    @property
    def mean(self):
        return self.to_std_parameters(self.natural_parameters)[0]

    @property
    def cov(self):
        return self.to_std_parameters(self.natural_parameters)[1]

    def moments(self):
        dim = self.dim[0] * self.dim[1]
        stats = self.expected_sufficient_statistics()
        W = stats[:dim].reshape(*self.dim)
        WW = stats[dim:].reshape(self.dim[0], self.dim[0])
        return W, WW

    def expected_value(self):
        mean, _ = self.to_std_parameters(self.natural_parameters)
        return mean

    def to_natural_parameters(self, mean, cov):
        prec = cov.inverse().contiguous()
        return torch.cat([
           (prec @ mean).contiguous().view(-1),
            -.5 * prec.view(-1)
        ])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        dim1, dim2 = self.dim
        dim = dim1 * dim2
        precision = - 2 *natural_parameters[dim:].view(dim1, dim1)
        cov = precision.inverse()
        mean = cov @ natural_parameters[:dim].view(dim1, dim2)
        return mean, cov

    def _expected_sufficient_statistics(self):
        mean, cov = self.to_std_parameters(self.natural_parameters)
        return torch.cat([
            mean.reshape(-1),
            (self.dim[1] * cov + mean @ mean.t()).reshape(-1)
        ])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, cov = self.to_std_parameters(natural_parameters)
        precision = cov.inverse()
        log_norm = - self.dim[1] * .5 * _logdet(precision)
        log_norm += .5 * torch.trace(mean.t() @ precision @  mean)
        return log_norm




class HierarchicalMatrixNormalPrior(MatrixNormalPrior):
    '''Matrix Normal prior with a hyper-prior over the diagional
    of the precision matrix.

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

    def __init__(self, mean, hyper_prior):
        '''
        Args:
            mean (``torch.Tensor[dim,dim]``)): Matrix mean.
            cov (``torch.tensor[1]``): Covariance matrix.
        '''
        self._dim = mean.shape
        self._hyper_prior = hyper_prior
        cov = torch.diag(1./hyper_prior.expected_value())
        nparams = self.to_natural_parameters(mean, cov)
        super(MatrixNormalPrior, self).__init__(nparams)

    @property
    def hyper_prior(self):
        return self._hyper_prior

    def to_natural_parameters(self, mean, cov):
        prec = torch.diag(self.hyper_prior.expected_value())
        return torch.cat([
           (prec @ mean).reshape(-1),
            -.5 * prec.view(-1)
        ])

    def _to_std_parameters(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        dim1, dim2 = self.dim
        dim = dim1 * dim2
        cov = torch.diag(1./self.hyper_prior.expected_value())
        mean = cov @ natural_parameters[:dim].reshape(dim1, dim2)
        return mean, cov

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        mean, _ = self.to_std_parameters(natural_parameters)
        precision = torch.diag(self.hyper_prior.expected_value())
        log_prec = self.hyper_prior.expected_sufficient_statistics()[self.dim[0]:]
        log_norm = - self.dim[1] * .5 * log_prec.sum()
        log_norm += .5 * torch.trace(mean.t() @ precision @  mean)
        return log_norm


__all__ = ['MatrixNormalPrior', 'HierarchicalMatrixNormalPrior']

