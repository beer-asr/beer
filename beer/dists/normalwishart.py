import abc
from dataclasses import dataclass
from functools import lru_cache
import math
import torch
from .basedist import ExponentialFamily


__all__ = ['NormalWishart', 'NormalWishartStdParams', 'JointNormalWishart',
           'JointNormalWishartStdParams']


@dataclass(init=False, eq=False, unsafe_hash=True)
class NormalWishartStdParams(torch.nn.Module):
    'Standard parameterization of the Normal-Wishart pdf.'

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
        # First we recover the dimension of the mean parameters (D).
        # Since the dimension of the natural parameters of the
        # Normal-Wishart is:
        #       l = len(natural_params)
        #       D^2 + D + 2 = l
        # we can find D by looking for the positive root of the above
        # polynomial which is given by:
        #       D = .5 * (-1 + sqrt(1 + 4 * l))
        dim = int(.5 * (-1 + math.sqrt(1 + 4 * len(natural_params))))

        np1 = natural_params[:dim]
        np2 = natural_params[dim: dim * (dim + 1)]
        np3 = natural_params[-2]
        np4 = natural_params[-1]
        scale = -2 * np3
        mean = np1 / scale
        scale_matrix = (-2 * np2.reshape(dim, dim) \
                        - scale * torch.ger(mean, mean)).inverse()
        dof = 2 * np4 + dim

        return cls(mean, scale, scale_matrix, dof)


class NormalWishart(ExponentialFamily):
    'Normal-Wishart pdf.'

    _std_params_def = {
        'mean': 'Mean of the Normal pdf.',
        'scale': 'scale of the precision of  the Normal pdf.',
        'scale_matrix': 'Scale matrix of the Wishart pdf.',
        'dof': 'Number degree of freedom of the Wishart pdf.',
    }

    @property
    def dim(self):
        '''Return a tuple with the dimension of the normal and the
        dimension of the Wishart: example (2, (2x2))

        '''
        return len(self.params.mean), tuple(self.params.scale_matrix.shape)

    @property
    def conjugate_sufficient_statistics_dim(self):
        d = self.dim[0]
        return 2 * d + d * (d - 1) // 2

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable mu (vector), S (positive definite
        matrix) the sufficient statistics of the Normal-Wishart are
        given by:

        stats = (
            S * mu,
            S,
            tr(S * mu * mu^T),
            ln |S|
        )

        For the standard parameters (m=mean, k=scale, W=W, v=dof)
        expectation of the sufficient statistics is given by:

        E[stats] = (
            v * W * m,
            v * W,
            (D/k) + tr(v * W * m * m^T),
            ( \sum_i psi(.5 * (v + 1 - i)) ) + D * ln 2 + ln |W|
        )

        Note: "tr" is the trace operator, "D" is the dimenion of "m"
            and "psi" is the "digamma" function.

        '''
        mean, scale, = self.params.mean, self.params.scale
        scale_matrix, dof = self.params.scale_matrix, self.params.dof
        dim = self.dim[0]
        idxs = torch.arange(1, dim + 1, dtype=mean.dtype, device=mean.device)
        L = torch.cholesky(scale_matrix, upper=False)
        logdet = 2 * torch.log(L.diag()).sum()
        mean_quad = torch.ger(mean, mean)
        exp_prec = dof * scale_matrix
        return torch.cat([
           exp_prec @ mean,
            exp_prec.reshape(-1),
            ((dim / scale) \
                + (exp_prec @ mean_quad).trace()).reshape(1),
            (torch.digamma(.5 * (dof + 1 - idxs)).sum() \
                + dim * math.log(2) + logdet).reshape(1)
        ])

    def expected_value(self):
        'The expected mean and the expected precision matrix.'
        return self.params.mean, self.params.dof * self.params.scale_matrix

    def log_norm(self):
        idxs = torch.arange(1, self.dim[0] + 1, dtype=self.params.mean.dtype,
                            device=self.params.mean.device)
        L = torch.cholesky(self.params.scale_matrix, upper=False)
        logdet = 2 * torch.log(L.diag()).sum()
        dim = self.dim[0]
        dof = self.params.dof
        scale = self.params.scale
        return .5 * dof * logdet + .5 * dof * dim * math.log(2) \
               + .25 * dim * (dim - 1) * math.log(math.pi) \
               + torch.lgamma(.5 * (dof + 1 - idxs)).sum() \
               - .5 * dim * torch.log(scale) \
               + .5 * dim * math.log(2 * math.pi)

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (m=mean, k=scale, W=W, v=dof) the natural
        parameterization is given by:

        nparams = (
            k * m ,
            -.5 * W^{-1} + k * m * m^T,
            -.5 * k,
            .5 * (v - D)
        )

        Note: "D" is the dimension of "m"

        Returns:
            ``torch.Tensor[D + D^2 + 2]``

        '''
        mean, scale, = self.params.mean, self.params.scale
        scale_matrix, dof = self.params.scale_matrix, self.params.dof
        dim = self.dim[0]
        return torch.cat([
            scale * mean,
            -.5 * (scale_matrix.inverse() \
                + scale * torch.ger(mean, mean)).reshape(-1),
            -.5 * scale.reshape(1),
            .5 * (dof - dim).reshape(1)
        ])

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)

    def sufficient_statistics_from_rvectors(self, rvecs):
        '''
        Real vector z = (u, v, w)
        \mu = u
        \Lambda = L L^T
        diag(L) = \exp(\frac{1}{2} v)
        tril(L) = w

        '''
        dim = self.dim[0]
        mean = rvecs[:, :dim]
        log_prec_L_diag = rvecs[:, dim:2*dim]
        prec_L_diag = (.5 * log_prec_L_diag).exp()   
        prec_L_offdiag = rvecs[:, 2*dim:] 

        # Build the precision matrices using the 
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
        

@dataclass(init=False, eq=False, unsafe_hash=True)
class JointNormalWishartStdParams(torch.nn.Module):
    means: torch.Tensor
    scales: torch.Tensor
    scale_matrix: torch.Tensor
    dof: torch.Tensor

    def __init__(self, means, scales, scale_matrix, dof):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('scales', scales)
        self.register_buffer('scale_matrix', scale_matrix)
        self.register_buffer('dof', dof)

    @classmethod
    def from_natural_parameters(cls, natural_params, ncomp):
        # First we recover the dimension of the mean parameters (D).
        # Since the dimension of the natural parameters of the
        # joint Normal-Wishart is:
        #       l -> len(natural_params)
        #       k -> Number of components
        #       D^2 + Dk + 1 + k = l
        # we can find D by looking for the positive root of the above
        # polynomial which is given by:
        #       D = .5 * (-k + sqrt(k^2 - 4 * (k + 1 - l)))
        l = len(natural_params)
        k = ncomp
        dim = int(.5 * (-k + math.sqrt(k**2 - 4 * (k + 1 - l))))

        np1s = natural_params[:ncomp * dim].reshape((ncomp, dim))
        np2 = natural_params[ncomp * dim: ncomp * dim + int(dim**2)]
        np2 = np2.reshape(dim, dim)
        np3s = natural_params[-(ncomp+1):-1]
        np4 = natural_params[-1]

        scales = -2 * np3s
        dof = 2 * np4 + dim + 1 - ncomp
        means = np1s / scales[:, None]
        quad_means = (scales[:, None] * means).t() @ means
        scale_matrix = torch.inverse(-2 * np2 - quad_means)

        return cls(means, scales, scale_matrix, dof)


class JointNormalWishart(ExponentialFamily):
    '''Set of Normal distributions sharing the same Wishart prior over
    the precision matrix.

    '''

    _std_params_def = {
        'means': 'Set of mean parameter.',
        'scales': 'Set of scaling of the precision matrix (for each Normal).',
        'scale_matrix': 'Scale matrix of the Wishart pdf.',
        'dof': 'Number degree of freedom of the Wishart pdf.',
    }

    @property
    def dim(self):
        '''Return a tuple ((K, D), D^2)' where K is the number of Normal
        and D is the dimension of their support.

        '''
        return (tuple(self.params.means.shape), self.params.means.shape[-1]**2)

    @property
    def conjugate_sufficient_statistics_dim(self):
        k, d = self.dim[0]
        return (k + 1) * d + d * (d - 1) // 2

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variables mu (set of vectors), S (positive
        definite matrix) the sufficient statistics of the joint
        Normal-Wishart are given by:

        stats = (
            S * mu_i,
            S,
            tr(S* mu * mu^T),
            ln(det(S))
        )

        For the standard parameters (m=mean, k=scale, W=scale_matrix,
        v=dof) expectation of the sufficient statistics is given by:

        E[stats] = (
            v * W * m_i,
            v * W,
            (D/k_i) + tr(v * W * m_i * m_i^T),
            ( \sum_i psi(.5 * (v + 1 - i)) ) + D * ln 2 + ln |W|
        )


        Note: "D" is the dimenion of "m", "tr" is the trace operation
            and "psi" is the "digamma" function.

.       '''
        ncomp, dim = self.dim[0]
        dtype, device = self.params.means.dtype, self.params.means.device
        precision = self.params.dof * self.params.scale_matrix
        L = torch.cholesky(self.params.scale_matrix, upper=False)
        logdet = 2 * torch.log(L.diag()).sum()
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        sum_digamma = torch.digamma(.5 * (self.params.dof + 1 - seq)).sum()
        quad_means = (self.params.means[:, :, None] @ self.params.means[:, None, :])
        quad_means = quad_means.reshape(ncomp, -1)
        vec_precision = precision.reshape(-1)
        return torch.cat([
            (self.params.means @ precision).view(-1),
            precision.reshape(-1),
            (dim / self.params.scales) + quad_means @ vec_precision,
            (sum_digamma + dim * math.log(2) + logdet).view(1)
        ])

    def expected_value(self):
        'Expected means and expected precision matrix.'
        return self.params.means, self.params.dof * self.params.scale_matrix

    def log_norm(self):
        dtype, device = self.params.means.dtype, self.params.means.device
        dim = self.dim[0][1]
        L = torch.cholesky(self.params.scale_matrix, upper=False)
        logdet = 2 * torch.log(L.diag()).sum()
        lognorm_prec = .5 * self.params.dof * logdet
        lognorm_prec += .5 * self.params.dof * dim * math.log(2)
        lognorm_prec += .25 * dim * (dim - 1) * math.log(math.pi)
        seq = torch.arange(1, dim + 1, 1, dtype=dtype, device=device)
        lognorm_prec += torch.lgamma(.5 * (self.params.dof + 1 - seq)).sum()
        lognorm = -.5 * dim  * torch.log(self.params.scales).sum()
        return lognorm + lognorm_prec

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        ncomp, dim = self.dim[0]
        inv_mean_prec = self.params.scale_matrix.inverse()
        means = self.params.means
        quad_means = (self.params.scales[:, None] * means).t() @ means
        return torch.cat([
            (self.params.scales[:, None] * means).reshape(-1),
            -.5 * (quad_means + inv_mean_prec).reshape(-1),
            -.5 * self.params.scales,
            .5 * (self.params.dof - dim - 1 + ncomp).view(1),
        ])

    def update_from_natural_parameters(self, natural_params):
        ncomp = self.dim[0][0]
        self.params = self.params.from_natural_parameters(natural_params, ncomp)

    def sufficient_statistics_from_rvectors(self, rvecs):
        '''
        Real vector z = (u, v, w)
        \mu = u
        \Lambda = L L^T
        diag(L) = \exp(\frac{1}{2} v)
        tril(L) = w

        '''
        k, dim = self.dim[0]
        means = rvecs[:, :k * dim].reshape(-1, k, dim)
        log_diag_prec = rvecs[:, k * dim:(k + 1) * dim]
        prec_L_diag = log_diag_prec.exp() 
        prec_L_offdiag = rvecs[:, (k + 1) * dim:] 

        # Build the precision matrices.
        arg = torch.arange(1, dim**2 + 1).view(dim, dim)
        tril_idxs = arg.tril(diagonal=-1).nonzero().t()
        diag_idxs = torch.arange(dim)
        L = torch.zeros(len(rvecs), dim, dim, dtype=rvecs.dtype, 
                        device=rvecs.device)
        L[:, tril_idxs[0], tril_idxs[1]] = prec_L_offdiag
        L[:, diag_idxs, diag_idxs] = (prec_L_diag).exp()
        prec_matrices = torch.matmul(L, L.permute(0, 2, 1))

        # Compute the product L^T mu mu^T L
        Lm = torch.matmul(means, L.permute(0, 2, 1))
        mPm = torch.sum(Lm * Lm, dim=-1)

        return torch.cat([
            torch.matmul(means, prec_matrices).reshape(-1, k * dim),
            prec_matrices.reshape(-1, dim ** 2),
            mPm,
            2 * log_diag_prec.sum(dim=-1).view(-1, 1),
            
        ], dim=-1)
