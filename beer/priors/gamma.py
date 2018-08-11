'''Implementation of the Dirichlet prior.'''

import torch
from .baseprior import ExpFamilyPrior


class GammaPrior(ExpFamilyPrior):
    '''Gamma distribution.

    parameters:
        a: shape
        b: rate

    natural parameters:
        eta1 = -b
        eta2 = a - 1

    sufficient statistics:
        T_1(x) = x
        T_2(x) = ln x

    '''

    def __init__(self, shape, rate):
        nparams = self.to_natural_parameters(shape, rate)
        super().__init__(nparams)

    @property
    def strength(self):
        self.natural_parameters[-1] + 1

    @strength.setter
    def strength(self, value):
        nparams = self.natural_parameters
        nparams[-1] = value - 1
        self.natural_parameters = nparams

    def to_std_parameters(self, natural_parameters):
        return  natural_parameters[1] + 1, -natural_parameters[0]

    def to_natural_parameters(self, shape, rate):
        return torch.cat([-rate.view(1), (shape - 1).view(1)])

    def expected_sufficient_statistics(self):
        shape, rate = self.to_std_parameters(self.natural_parameters)
        return torch.cat([(shape / rate).view(1),
                          (torch.digamma(shape) - torch.log(rate)).view(1)])

    def expected_value(self):
        shape, rate = self.to_std_parameters(self.natural_parameters)
        return shape / rate

    def log_norm(self, natural_parameters):
        shape, rate = self.to_std_parameters(natural_parameters)
        return torch.lgamma(shape) - shape * torch.log(rate)


__all__ = ['GammaPrior']
