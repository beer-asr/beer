import abc
from dataclasses import dataclass
import math
import torch

from .basemodel import Model
from .parameters import ConjugateBayesianParameter
from ..dists import NormalWishart, NormalWishartStdParams
from ..dists import NormalGamma, NormalGammaStdParams
from ..dists import IsotropicNormalGamma, IsotropicNormalGammaStdParams


__all__ = ['Normal']


# Error raised when attempting to create a Normal model object with an
# unknwown type of covariance matrix.
class UnknownCovarianceType(Exception): pass

########################################################################
# Helper to build the default parameters.

# Return a full covariance matrix whether the user has specified a 
# scalar, a diagonal or a full matrix.
def _full_cov(cov, tensorconf):
    if len(cov.shape) == 1 and cov.shape[0] == 1:
        return cov * torch.eye(len(mean), **tensorconf)
    elif len(cov.shape) == 1:
        return cov.diag()
    return cov

def _default_fullcov_param(mean, cov, prior_strength, tensorconf):
    cov = _full_cov(cov, tensorconf)
    scale = torch.tensor(prior_strength, **tensorconf)
    dof = torch.tensor(prior_strength + len(mean) - 1, **tensorconf)
    scale_matrix = cov.inverse() / dof
    params = NormalWishartStdParams(mean, scale, scale_matrix, dof)
    prior = NormalWishart(params)
    params = NormalWishartStdParams(mean, scale, scale_matrix, dof)
    posterior = NormalWishart(params)
    return ConjugateBayesianParameter(prior, posterior)

def _default_diagcov_param(mean, cov, prior_strength, tensorconf):
    cov = _full_cov(cov, tensorconf)
    variance = cov.diag()
    scale = torch.tensor(prior_strength, **tensorconf)
    shape = torch.tensor(prior_strength, **tensorconf)
    rates = prior_strength * variance
    params = NormalGammaStdParams(mean, scale, shape, rates)
    prior = NormalGamma(params)
    params = NormalGammaStdParams(mean, scale, shape, rates)
    posterior = NormalGamma(params)
    return ConjugateBayesianParameter(prior, posterior)

def _default_isocov_param(mean, cov, prior_strength, tensorconf):
    cov = _full_cov(cov, tensorconf)
    variance = cov.diag().max()
    scale = torch.tensor(prior_strength, **tensorconf)
    shape = torch.tensor(prior_strength, **tensorconf)
    rate =  prior_strength * variance
    params = IsotropicNormalGammaStdParams(mean, scale, shape, rate)
    prior = IsotropicNormalGamma(params)
    params = IsotropicNormalGammaStdParams(mean, scale, shape, rate)
    posterior = IsotropicNormalGamma(params)
    return ConjugateBayesianParameter(prior, posterior)

_default_param = {
    'full': _default_fullcov_param,
    'diagonal': _default_diagcov_param,
    'isotropic': _default_isocov_param,
}

########################################################################

class Normal(Model):
    '''Normal model with prior over the mean and variance parameter.

    Attributes:
        mean_precision: Joint mean/precision parameter.

    '''

    @classmethod
    def create(cls, mean, cov, prior_strength=1., cov_type='full'):
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
        if cov_type not in cov_type:
            raise UnknownCovarianceType('Unknown covariance type: ' \
                                        f'"{cov_type}"')

        tensorconf = {'dtype': mean.dtype, 'device': mean.device, 
                      'requires_grad': False}
        mean = mean.detach()
        cov = cov.detach()
        makeparam = _default_param[cov_type]
        return cls(makeparam(mean, cov, prior_strength, tensorconf))

    def __init__(self, mean_precision):
        super().__init__()
        self.mean_precision = mean_precision

    ####################################################################
    # The following properties are exposed only for plotting/debugging
    # purposes.

    @property
    def mean(self):
        return self.mean_precision.value()[0]

    @property
    def cov(self):
        precision = self.mean_precision.value()[1]
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

    def sufficient_statistics(self, data):
        return self.mean_precision.sufficient_statistics(data)

    def mean_field_factorization(self):
        return [[self.mean_precision]]

    def expected_log_likelihood(self, stats):
        dim = self.mean_precision.prior.dim[0]
        nparams = self.mean_precision.natural_form()
        return (stats * nparams[None]).sum(dim=-1) \
               -.5 * dim * math.log(2 * math.pi)

    def accumulate(self, stats, parent_msg=None):
        return {self.mean_precision: stats.sum(dim=0)}
