import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily
from .basedist import ConjugateLikelihood


__all__ = ['Dirichlet', 'DirichletStdParams', 'CategoricalLikelihood']


class CategoricalLikelihood(ConjugateLikelihood):

    __slots__ = 'dim'

    def __init__(self, dim):
        self.dim = dim

    def __repr__(self):
        return f'{self.__class__.__qualname__}(dim={self.dim})'

    @property
    def sufficient_statistics_dim(self):
        return self.dim - 1

    @staticmethod
    def log_norm(nparams):
        # Stable implementation of the log-normalizer of a categorical
        # equivalent of the naive implementation:
        #   lognorm = torch.log(1 + rvectors.exp())
        return torch.nn.functional.softplus(torch.logsumexp(nparams, dim=-1))

    @staticmethod
    def parameters_from_pdfvector(pdfvec):
        'Return the parameters of the pdf vector.' 
        retval = torch.zeros_like(pdfvec, requires_grad=False)
        lnorm = CategoricalLikelihood.log_norm(pdfvec[:-1])
        remainder = (-lnorm).exp()
        retval[:-1] = pdfvec[:-1].exp() * remainder
        retval[-1] = remainder
        return retval

    @staticmethod
    def pdfvectors_from_rvectors(rvecs):
        '''
        Returns:

            (x, -A(x)) 

        with:
             A(x) = ln( 1 + exp( sum_i^{D-1} (x_i) ) )

        '''
        lnorm = CategoricalLikelihood.log_norm(rvecs)
        return torch.cat([rvecs, -lnorm.view(-1, 1)], dim=-1)

    
@dataclass(init=False, eq=False, unsafe_hash=True)
class DirichletStdParams(torch.nn.Module):
    'Standard parameterization of the Dirichlet pdf.'

    concentrations: torch.Tensor

    def __init__(self, concentrations):
        super().__init__()
        self.register_buffer('concentrations', concentrations)

    @classmethod
    def from_natural_parameters(cls, natural_params):
        return cls(natural_params + 1)


class Dirichlet(ExponentialFamily):
    'Dirichlet Distribution.'

    _std_params_def = {
        'concentrations': 'Concentrations parameter.'
    }

    def conjugate(self):
        return CategoricalLikelihood(self.dim)

    @property
    def dim(self):
        return len(self.params.concentrations)

    def expected_sufficient_statistics(self):
        '''Expected sufficient statistics given the current
        parameterization.

        For the random variable p (vector of probabilities)
        the sufficient statistics of the Dirichlet are
        given by:

        stats = (
            ln(p)
        )

        For the standard parameters (a=concentrations) expectation of
        the sufficient statistics is given by:

        E[stats] = (
            psi(a) - psi(\sum_i a_i)
        )

        Note: ""D" is the dimenion of "m"
            and "psi" is the "digamma" function.

        '''
        return torch.digamma(self.params.concentrations) \
               - torch.digamma(self.params.concentrations.sum())

    def expected_value(self):
        'Expected distribution p.'
        return self.params.concentrations / self.params.concentrations.sum()

    def log_norm(self):
        return torch.lgamma(self.params.concentrations).sum() \
               - torch.lgamma(self.params.concentrations.sum())

    # TODO
    def sample(self, nsamples):
        raise NotImplementedError

    def natural_parameters(self):
        '''Natural form of the current parameterization. For the
        standard parameters (a=concentrations) the natural
        parameterization is given by:

        nparams = (
            a - 1,
        )

        Returns:
            ``torch.Tensor[D]`` where D is the dimension of the support.

        '''
        return self.params.concentrations - 1

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)
