import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily

__all__ = ['NormalWishart', 'NormalWishartStdParams']


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
        return len(self.mean), tuple(self.scale_matrix.shape)

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
        idxs = torch.arange(1, self.dim[0] + 1, dtype=self.mean.dtype,
                            device=self.mean.device)
        L = torch.cholesky(self.scale_matrix, upper=False)
        logdet = 2 * torch.log(L.diag()).sum()
        mean_quad = torch.ger(self.mean, self.mean)
        exp_prec = self.dof * self.scale_matrix
        return torch.cat([
           exp_prec @ self.mean,
            exp_prec.reshape(-1),
            ((self.dim[0] / self.scale) \
                + (exp_prec @ mean_quad).trace()).reshape(1),
            (torch.digamma(.5 * (self.dof + 1 - idxs)).sum() \
                + self.dim[0] * math.log(2) + logdet).reshape(1)
        ])

    def expected_value(self):
        'The expected mean and the expected precision matrix.'
        return self.mean, self.dof * self.scale_matrix

    def log_norm(self):
        idxs = torch.arange(1, self.dim[0] + 1, dtype=self.mean.dtype,
                            device=self.mean.device)
        L = torch.cholesky(self.scale_matrix, upper=False)
        logdet = 2 * torch.log(L.diag()).sum()
        return .5 * self.dof * logdet + .5 * self.dof * self.dim[0] * math.log(2) \
               + .25 * self.dim[0] * (self.dim[0] - 1) * math.log(math.pi) \
               + torch.lgamma(.5 * (self.dof + 1 - idxs)).sum() \
               - .5 * self.dim[0] * torch.log(self.scale) \
               + .5 * self.dim[0] * math.log(2 * math.pi)

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
        return torch.cat([
            self.scale * self.mean,
            -.5 * (self.scale_matrix.inverse() \
                + self.scale * torch.ger(self.mean, self.mean)).reshape(-1),
            -.5 * self.scale.reshape(1),
            .5 * (self.dof - self.dim[0]).reshape(1)
        ])

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)

