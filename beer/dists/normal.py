import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihood


__all__ = ['NormalFullCovariance', 'NormalFullCovarianceStdParams']


@dataclass(init=False, eq=False, unsafe_hash=True)
class NormalFullCovarianceStdParams(torch.nn.Module):
    '''Standard parameterization of the Normal pdf with full
    covariance matrix.
    '''
    mean: torch.Tensor
    cov: torch.Tensor

    def __init__(self, mean, cov):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        npsize = natural_params.shape
        if len(npsize) == 1:
            natural_params = natural_params.view(1, -1)

        # First we recover the dimension of the mean parameters (D).
        # Since the dimension of the natural parameters of the
        # Normal-Wishart is:
        #       l = natural_params.shape[-1]
        #       D^2 + D = l
        # we can find D by looking for the positive root of the above
        # polynomial which is given by:
        #       D = .5 * (-1 + sqrt(1 + 4 * l))
        l = natural_params.shape[-1]
        dim = int(.5 * (-1 + math.sqrt(1 + 4 * l)))

        np1 = natural_params[:, :dim]
        np2 = natural_params[:, dim: dim * (dim + 1)]
        cov = np2.inverse()
        mean = cov * np1

        if len(npsize) == 1:
            return cls(mean.view(-1), cov.view(dim, dim))
        return cls(mean.view(-1, dim), cov.view(-1, dim, dim))


class NormalFullCovariance(ExponentialFamily):
    _std_params_def = {
        'mean': 'Mean parameter.',
        'cov': 'Covariance matrix.',
    }

    _std_params_cls = NormalFullCovarianceStdParams

    def __len__(self):
        paramshape = self.params.mean.shape
        return 1 if len(paramshape) <= 1 else paramshape[0]

    @property
    def dim(self):
        return self.params.mean.shape[-1]

    def conjugate(self):
        raise NotImplementedError

    def forward(self, stats, pdfwise=False):
        nparams = self.natural_parameters()
        mean = self.params.mean
        cov = self.params.cov
        size = mean.shape
        dim = self.dim
        if len(size) <= 1:
            mean = mean.view(1, -1)
            nparams = nparams.view(1, -1)

        # Get the precision matrix from the natural parameters not
        # to inverse the covariance matrix once more time.
        prec = nparams[:, dim: dim * (dim + 1)].reshape(-1, dim, dim)

        # Log determinant of all the covariance matrix.
        L = torch.cholesky(cov, upper=False)
        logdet = 2 * torch.log(L[:, range(dim), range(dim)]).sum(dim=-1)

        # Quadratic term of the log-normalizer: -.5 * mu^T S mu
        Sm = nparams[:, :dim]
        mSm = (Sm * mean).sum(dim=-1)
        lnorm = .5 * (logdet + mSm)
        log_basemeasure = -.5 * (dim * math.log(2 * math.pi))

        if pdfwise:
            return torch.sum(nparams * stats, dim=-1) - lnorm \
                   + log_basemeasure
        retval = nparams @ stats.t() - lnorm[:, None] + log_basemeasure
        if len(size) <= 1:
            return retval.reshape(-1)
        return retval

    def sufficient_statistics(self, data):
        data_quad = (data[:, :, None] * data[:, None, :])
        data_quad = data_quad.reshape(len(data), -1)
        return torch.cat([data, -.5 * data_quad], dim=-1)

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable x (vector)the sufficient statistics of
        the Normal with diagonal covariance matrix are given by:

        stats = (
            x,
            x**2,
        )

        For the standard parameters (m=mean, s=diagonal of the cov.
        matrix) the expectation of the sufficient statistics is
        given by:

        E[stats] = (
            m,
            s + m**2
        )

        '''
        mean, cov = self.params.mean, self.params.cov
        size = mean.shape
        dim = self.dim
        if len(size) <= 1:
            mean = mean.view(1, -1)
            cov = cov.view(1, dim, dim)
        mean_quad = mean[:, :, None] * mean[:, None, :]
        retval = torch.cat([
            mean,
            -.5 * (cov +  mean_quad).reshape(len(mean), -1)
        ], dim=-1)
        if len(size) <= 1:
            return retval.view(-1)
        return retval

    def expected_value(self):
        return self.params.mean

    def log_norm(self):
        mean, cov = self.params.mean, self.params.cov
        size = mean.shape
        dim = self.dim
        if len(size) <= 1:
            mean = mean.view(1, -1)
            cov = cov.view(1, dim, dim)

        L = torch.cholesky(cov, upper=False)
        logdet = 2 * torch.log(L[:, range(dim), range(dim)]).sum(dim=-1)
        prec = cov.inverse()
        Sm = torch.matmul(prec, mean[:, :, None]).view(-1, dim)
        mSm = (Sm * mean).sum(dim=-1)
        log_base_measure = .5 * dim * math.log(2 * math.pi)
        return .5 * (logdet + mSm) + log_base_measure

    def sample(self, nsamples):
        mean, cov = self.params.mean, self.params.cov
        size = mean.shape
        dim = self.dim
        if len(size) <= 1:
            mean = mean.view(1, -1)
            cov = cov.view(1, dim, dim)
        L = torch.cholesky(cov, upper=False)
        noise = torch.randn(mean.shape[0], nsamples, mean.shape[1],
                            dtype=mean.dtype, device=mean.device)
        retval = mean[:, None, :] + torch.matmul(noise, L.permute(0, 2, 1))
        if len(size) <= 1:
            return retval.reshape(nsamples, -1)
        return retval

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (m=mean, s=diagonal of the cov. matrix) the
        natural parameterization is given by:

        nparams = (
            s^-1 * m ,
            -.5 * s^1
        )

        Returns:
            ``torch.Tensor[2 * D]``

        '''
        mean, cov = self.params.mean, self.params.cov
        size = len(mean.shape) if len(mean.shape) > 0 else 1
        dim = mean.shape[-1]
        if size == 1:
            mean = mean.view(1, -1)
            cov = cov.view(1, dim, dim)
        prec = cov.inverse()
        Sm = torch.matmul(prec, mean[:, :, None]).view(-1, dim)
        retval = torch.cat([Sm, prec.reshape(len(mean), -1)], dim=-1)
        if size == 1:
            return retval.view(-1)
        return retval

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)

