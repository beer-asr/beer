import abc
from dataclasses import dataclass
import math
import torch
from .basedist import ExponentialFamily


__all__ = ['Dirichlet', 'DirichletStdParams']


@dataclass(init=False, eq=False, unsafe_hash=True)
class DirichletStdParams(torch.nn.Module):
    'Standard parameterization of the Dirichlet pdf.'

    concentrations: torch.Tensor

    def __init__(self, concentrations):
        super.__init__()
        self.register_buffer('concentrations', concentrations)

    @classmethod
    def from_natural_parameters(self, natural_params):
        return natural_parameters + 1


class DirichletPrior(ExponentialFamily):
    'Dirichlet Distribution.'

    _std_params_def = {
        'concentrations': 'Concentrations parameter.'
    }

    @property
    def dim(self):
        return len(self.concentrations)

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
        return torch.digamma(self.concentrations) \
               - torch.digamma(self.concentrations.sum())

    def expected_value(self):
        'Expected distribution p.'
        return self.concentrations / self.concentrations.sum()

    def log_norm(self):
        return torch.lgamma(self.concentrations).sum() \
               - torch.lgamma(self.concentrations.sum())

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
        return self.concentrations - 1

    def update_from_natural_parameters(self, natural_params):
        self.params = self.params.from_natural_parameters(natural_params)

