import abc
import math
import torch

from .basemodel import Model
from .parameters import ConjugateBayesianParameter
from ..dists import NormalWishart, NormalWishartStdParams
from ..dists import NormalGamma, NormalGammaStdParams
from ..dists import IsotropicNormalGamma, IsotropicNormalGammaStdParams


__all__ = ['Normal', 'NormalIsotropicCovariance', 'NormalDiagonalCovariance',
           'NormalFullCovariance']


# Error raised when attempting to create a Normal model object with an
# unknwown type of covariance matrix.
class UnknownCovarianceType(Exception): pass


class Normal(Model):
    '''Normal model with prior over the mean and variance parameter.

    Attributes:
        mean: Mean parameter.
        cov: Covariance parameter.

    '''

    @staticmethod
    def create(mean, cov, prior_strength=1., cov_type='full'):
        '''Create a Normal model.

        Args:
            mean (``torch.Tensor[dim]``): Initial mean of the model.
            cov (``torch.Tensor[dim, dim]`` or ``torch.Tensor[dim]`` or
                scalar):
                Initial covariance matrix. Can be specified as a
                dense/diagonal matrix or a scalar.
            cov_type (str): Type of the covariance matrix. Can be
                "full", "diagonal", or "isotropic".

        Returns:
            :any:`Normal`

        '''
        if cov_type not in cov_types:
            raise UnknownCovarianceType('Unknown covariance type: ' \
                                        f'"{cov_type}"')

        # Ensure the covariance is full.
        if len(cov.shape) == 1:
            if cov.shape[0] == 1:
                dtype, device = mean.dtype, mean.device
                full_cov = cov * torch.eye(len(mean), dtype=dtype, device=device)
            else:
                full_cov = cov.diag()
        else:
            full_cov = cov

        return cov_types[cov_type](mean, full_cov, prior_strength)

    def __init__(self, prior, posterior):
        super().__init__()
        self.mean_precision = ConjugateBayesianParameter(prior, posterior)

    ####################################################################
    # The following properties are exposed only for plotting/debugging
    # purposes.

    @property
    def expected_mean(self):
        return self.mean_precision.posterior.expected_value()[0]

    @property
    def expected_cov(self):
        precision = self.mean_precision.posterior.expected_value()[1]
        if len(precision.shape) == 2:
            # Full covariance matrix.
            return precision.inverse()
        elif len(precision.shape) == 1 and precision.shape[0] > 1:
            # Diagonal covariance matrix.
            return (1. / precision).diag()
        # Isotropic covariance matrix.
        dim = self.mean_precision.prior.dim[0]
        I = torch.eye(dim, dtype=precision.dtype, device=precision.device)
        return (1. / precision) * I

    ####################################################################

    def mean_field_factorization(self):
        return [[self.mean_precision]]

    def expected_log_likelihood(self, stats):
        dim = self.mean_precision.prior.dim[0]
        nparams = self.mean_precision.expected_natural_parameters()
        return (stats * nparams[None]).sum(dim=-1) \
               -.5 * dim * math.log(2 * math.pi)

    def accumulate(self, stats, parent_msg=None):
        return {self.mean_precision: stats.sum(dim=0)}


class NormalIsotropicCovariance(Normal):
    '''Normal model with isotropic covariance matrix.'''

    @classmethod
    def create(cls, mean, cov, prior_strength=1.):
        variance = cov.diag().max()
        dtype, device = mean.dtype, mean.device
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate =  prior_strength * variance
        params = IsotropicNormalGammaStdParams(
            mean.clone().detach(),
            scale.clone().detach(),
            shape.clone().detach(),
            rate.clone().detach()
        )
        prior = IsotropicNormalGamma(params)
        params = IsotropicNormalGammaStdParams(
            mean.clone().detach(),
            scale.clone().detach(),
            shape.clone().detach(),
            rate.clone().detach()
        )
        posterior = IsotropicNormalGamma(params)
        return cls(prior, posterior)

    @staticmethod
    def sufficient_statistics(data):
        dim, dtype, device = data.shape[1], data.dtype, data.device
        return torch.cat([
            data,
            -.5 * torch.sum(data**2, dim=-1).reshape(-1, 1),
            -.5 * torch.ones(len(data), 1, dtype=dtype, device=device),
            .5 * dim * torch.ones(len(data), 1, dtype=dtype, device=device),
        ], dim=-1)



class NormalDiagonalCovariance(Normal):
    '''Normal model with diagonal covariance matrix.'''

    @classmethod
    def create(cls, mean, cov, prior_strength=1.):
        variance = cov.diag()
        dtype, device = mean.dtype, mean.device
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rates = prior_strength * variance
        params = NormalGammaStdParams(
            mean.clone().detach(),
            scale.clone().detach(),
            shape.clone().detach(),
            rates.clone().detach()
        )
        prior = NormalGamma(params)

        params = NormalGammaStdParams(
            mean.clone().detach(),
            scale.clone().detach(),
            shape.clone().detach(),
            rates.clone().detach()
        )
        posterior = NormalGamma(params)
        return cls(prior, posterior)

    @staticmethod
    def sufficient_statistics(data):
        dtype, device = data.dtype, data.device
        return torch.cat([
            data,
            -.5 * data**2,
            -.5 * torch.ones(len(data), 1, dtype=dtype, device=device),
            .5 * torch.ones(len(data), 1, dtype=dtype, device=device),
        ], dim=-1)


class NormalFullCovariance(Normal):
    '''Normal model with full covariance matrix.'''

    @classmethod
    def create(cls, mean, cov, prior_strength=1.):
        dtype, device = mean.dtype, mean.device
        scale = torch.tensor(prior_strength, dtype=dtype, device=device)
        dof = torch.tensor(prior_strength + len(mean) - 1, dtype=dtype, device=device)
        scale_matrix = cov.inverse() / dof
        params = NormalWishartStdParams(
            mean.clone().detach(),
            scale.clone().detach(),
            scale_matrix.clone().detach(),
            dof.clone().detach()
        )
        prior = NormalWishart(params)

        params = NormalWishartStdParams(
            mean.clone().detach(),
            scale.clone().detach(),
            scale_matrix.clone().detach(),
            dof.clone().detach()
        )
        posterior = NormalWishart(params)
        return cls(prior, posterior)

    @staticmethod
    def sufficient_statistics(data):
        dtype, device = data.dtype, data.device
        data_quad = data[:, :, None] * data[:, None, :]
        return torch.cat([
            data,
            -.5 * data_quad.reshape(len(data), -1),
            -.5 * torch.ones(data.size(0), 1, dtype=dtype, device=device),
            .5 * torch.ones(data.size(0), 1, dtype=dtype, device=device),
        ], dim=-1)


# Different type of covariance and their respective constructor.
cov_types = {
    'full': NormalFullCovariance.create,
    'diagonal': NormalDiagonalCovariance.create,
    'isotropic': NormalIsotropicCovariance.create
}

