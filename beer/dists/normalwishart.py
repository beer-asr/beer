import abc
from dataclasses import dataclass
from functools import lru_cache
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihood


__all__ = ['NormalLikelihood', 'NormalWishart', 'NormalWishartStdParams']


# Helper function to compute the trace of a batch of pair of
# square matrices.
def _batch_trace(As, Bs, keepdim=False):
    n = len(As)
    return (As.reshape(n, -1) * Bs.reshape(n, -1)).sum(dim=-1, keepdim=keepdim)


@dataclass
class NormalLikelihood(ConjugateLikelihood):
    dim: int

    def sufficient_statistics_dim(self, zero_stats=True):
        d = self.dim
        zero_stats_dim = 2 if zero_stats else 0
        return 2 * d + d * (d - 1) // 2 + zero_stats_dim

    @staticmethod
    def sufficient_statistics(data):
        dtype, device = data.dtype, data.device
        data_quad = data[:, :, None] * data[:, None, :]
        return torch.cat([
            data,
            -.5 * data_quad.reshape(len(data), -1),
            -.5 * torch.ones(data.shape[0], 1, dtype=dtype, device=device),
            .5 * torch.ones(data.shape[0], 1, dtype=dtype, device=device),
        ], dim=-1)

    def parameters_from_pdfvector(self, pdfvec):
        size = pdfvec.shape
        if len(size) == 1:
            pdfvec = pdfvec.view(1, -1)
        dim = self.dim
        precision = pdfvec[:, dim:dim + dim ** 2].reshape(-1, dim, dim)
        cov = precision.inverse()
        mean = torch.matmul(cov, pdfvec[:, :dim, None])
        if len(size) == 1:
            return mean.view(-1), precision.view(dim, dim)
        return mean.view(-1, dim), precision.view(-1, dim, dim)

    def pdfvectors_from_rvectors(self, rvecs):
        '''
        Real vector z = (u, v, w)
        \mu = u
        \Lambda = L L^T
        diag(L) = \exp(\frac{1}{2} v)
        tril(L) = w

        '''
        dim = self.dim
        mean = rvecs[:, :dim]
        log_prec_L_diag = rvecs[:, dim:2*dim]
        prec_L_diag = (.5 * log_prec_L_diag).exp()
        prec_L_offdiag = rvecs[:, 2*dim:]

        # Build the precision matrices.
        arg = torch.arange(1, dim**2 + 1).view(dim, dim)
        tril_idxs = arg.tril(diagonal=-1).nonzero().t()
        diag_idxs = torch.arange(dim)
        L = torch.zeros(len(rvecs), dim, dim, dtype=rvecs.dtype,
                        device=rvecs.device)
        L[:, tril_idxs[0], tril_idxs[1]] = prec_L_offdiag
        L[:, diag_idxs, diag_idxs] = prec_L_diag
        prec_matrices = torch.matmul(L, L.permute(0, 2, 1))

        # Compute the product L^T mu mu^T L
        Lm = torch.matmul(L.permute(0, 2, 1), mean[:, :, None]).reshape(-1, dim)
        mPm = torch.matmul(Lm[:, None, :], Lm[:, :, None])

        return torch.cat([
            torch.matmul(prec_matrices, mean[:, :, None]).reshape(-1, dim),
            prec_matrices.reshape(-1, dim ** 2),
            mPm.reshape(-1, 1),
            log_prec_L_diag.sum(dim=-1).view(-1, 1),
        ], dim=-1)

    def __call__(self, pdfvecs, stats):
        if len(pdfvecs.shape) == 1:
            pdfvecs = pdfvecs.view(1, -1)
        log_basemeasure = -.5 * self.dim * math.log(2 * math.pi)
        return stats @ pdfvecs.t() + log_basemeasure


@dataclass(init=False, unsafe_hash=True)
class NormalWishartStdParams(torch.nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor
    scale_matrix: torch.Tensor
    dof: torch.Tensor

    def __init__(self, mean, scale, scale_matrix, dof):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('scale_matrix', scale_matrix)
        self.register_buffer('dof', dof)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        npsize = natural_params.shape
        if len(npsize) == 1:
            natural_params = natural_params.view(1, -1)

        # First we recover the dimension of the mean parameters (D).
        # Since the dimension of the natural parameters of the
        # Normal-Wishart is:
        #       l = natural_params.shape[-1] - 2
        #       D^2 + D = l
        # we can find D by looking for the positive root of the above
        # polynomial which is given by:
        #       D = .5 * (-1 + sqrt(1 + 4 * l))
        l = natural_params.shape[-1] - 2
        dim = int(.5 * (-1 + math.sqrt(1 + 4 * l)))

        np1 = natural_params[:, :dim]
        np2 = natural_params[:, dim: dim * (dim + 1)]
        np3 = natural_params[:, -2]
        np4 = natural_params[:, -1]
        scale = -2 * np3
        mean = np1 / scale[:, None]
        mean_quad = mean[:, :, None] * mean[:, None, :]
        scale_matrix = (-2 * np2.reshape(-1, dim, dim) \
                        - scale[:, None, None] * mean_quad).inverse()
        dof = 2 * np4 + dim

        if len(npsize) == 1:
            return cls(mean.view(-1), scale.view(1),
                       scale_matrix.view(dim, dim), dof.view(1))
        return cls(mean, scale.view(-1, 1), scale_matrix.view(-1, dim, dim),
                  dof.view(-1, 1))


class NormalWishart(ExponentialFamily):
    _std_params_def = {
        'mean': 'Mean of the Normal pdf.',
        'scale': 'scale of the precision of  the Normal pdf.',
        'scale_matrix': 'Scale matrix of the Wishart pdf.',
        'dof': 'Number degree of freedom of the Wishart pdf.',
    }

    _std_params_cls = NormalWishartStdParams

    def __len__(self):
        paramshape = self.params.mean.shape
        return 1 if len(paramshape) <= 1 else paramshape[0]

    @property
    def dim(self):
        '''Return a tuple with the dimension of the normal and the
        dimension of the Wishart: example (2, (2x2))

        '''
        dim = self.params.mean.shape[-1]
        return (*self.params.mean.shape, (dim, dim))

    def conjugate(self):
        return NormalLikelihood(self.params.mean.shape[-1])

    def expected_sufficient_statistics(self):
        '''
        stats = (
            S * mu,
            S,
            tr(S * mu * mu^T),
            ln |S|
        )

        E[stats] = (
            v * W * m,
            v * W,
            (D/k) + tr(v * W * m * m^T),
            ( \sum_i psi(.5 * (v + 1 - i)) ) + D * ln 2 + ln |W|
        )
        '''
        mean, scale, = self.params.mean, self.params.scale
        scale_matrix, dof = self.params.scale_matrix, self.params.dof
        dim = mean.shape[-1]
        mean_size = mean.shape if len(mean.shape) > 0 else 1
        if len(mean_size) == 1:
            mean, scale = mean.view(1, -1), scale.view(1, 1)
            scale_matrix, dof = scale_matrix.view(1, dim, dim), dof.view(1, 1)

        idxs = torch.arange(1, dim + 1, dtype=mean.dtype, device=mean.device)
        L = torch.cholesky(scale_matrix, upper=False)
        logdet = 2 * torch.log(L[:, range(dim), range(dim)]).sum(dim=-1,
                                                                 keepdim=True)
        mean_quad = mean[:, :, None] * mean[:, None, :]
        exp_prec = dof[:, :, None] * scale_matrix
        retval = torch.cat([
            torch.matmul(exp_prec, mean[:, :, None]).reshape(-1, dim),
            exp_prec.reshape(-1, dim ** 2),
            (dim / scale) + _batch_trace(exp_prec, mean_quad, keepdim=True),
            (torch.digamma(.5 * (dof + 1 - idxs)).sum(dim=-1, keepdim=True) \
                + dim * math.log(2) + logdet)
        ], dim=-1)

        if len(mean_size) == 1:
            return retval.view(-1)
        return retval

    def expected_value(self):
        'The expected mean and the expected precision matrix.'
        if len(self.params.mean.shape) == 1:
            return self.params.mean, self.params.dof * self.params.scale_matrix
        return self.params.mean, \
               self.params.dof[:, :, None] * self.params.scale_matrix

    def log_norm(self):
        mean, scale, = self.params.mean, self.params.scale
        scale_matrix, dof = self.params.scale_matrix, self.params.dof
        dim = mean.shape[-1]
        mean_size = mean.shape if len(mean.shape) > 0 else 1
        if len(mean_size) == 1:
            mean, scale = mean.view(1, -1), scale.view(1, 1)
            scale_matrix, dof = scale_matrix.view(1, dim, dim), dof.view(1, 1)

        idxs = torch.arange(1, dim + 1, dtype=mean.dtype, device=mean.device)
        L = torch.cholesky(scale_matrix, upper=False)
        logdet = 2 * torch.log(L[:, range(dim), range(dim)]).sum(dim=-1,
                                                                 keepdim=True)
        return (.5 * dof * logdet + .5 * dof * dim * math.log(2) \
               + .25 * dim * (dim - 1) * math.log(math.pi) \
               + torch.lgamma(.5 * (dof + 1 - idxs)).sum(dim=-1, keepdim=True) \
               - .5 * dim * torch.log(scale) \
               + .5 * dim * math.log(2 * math.pi)).sum(dim=-1)

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''
        nparams = (
            k * m ,
            -.5 * W^{-1} + k * m * m^T,
            -.5 * k,
            .5 * (v - D)
        )
        '''
        mean, scale, = self.params.mean, self.params.scale
        scale_matrix, dof = self.params.scale_matrix, self.params.dof
        dim = mean.shape[-1]
        mean_size = mean.shape if len(mean.shape) > 0 else 1
        if len(mean_size) == 1:
            mean, scale = mean.view(1, -1), scale.view(1, 1)
            scale_matrix, dof = scale_matrix.view(1, dim, dim), dof.view(1, 1)
        quad_mean = mean[:, :, None] * mean[:, None, :]
        retval = torch.cat([
            scale * mean,
            -.5 * (scale_matrix.inverse() \
                + scale[:, :, None] * quad_mean).reshape(-1, dim**2),
            -.5 * scale.reshape(-1, 1),
            .5 * (dof - dim).reshape(-1, 1)
        ], dim=-1)

        if len(mean_size) == 1:
            return retval.view(-1)
        return retval

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)

