'''Implementation of the Dirichlet prior.'''

import torch
from .baseprior import ExpFamilyPrior


class DirichletPrior(ExpFamilyPrior):
    '''Dirichlet Distribution.

    parameters:
        alpha[k] > 0

    natural parameters:
        eta[k] = alphas[k] - 1

    sufficient statistics:
        T(x) = ln x

    '''

    def __init__(self, alphas):
        nparams = self.to_natural_parameters(alphas)
        super().__init__(nparams)

    @property
    def strength(self):
        alphas = self.to_std_parameters(self.natural_params)
        mean = alphas / alphas.sum()
        return alphas[0] / mean[0]

    @strength.setter
    def strength(self, value):
        alphas = self.to_std_parameters(self.natural_parameters)
        mean = alphas / alphas.sum()
        new_alphas = value * mean
        self.natural_parameters = self.to_natural_parameters(new_alphas)

    def to_std_parameters(self, natural_params):
        return natural_params + 1

    def to_natural_parameters(self, std_params):
        return std_params - 1

    def expected_sufficient_statistics(self):
        alphas = self.to_std_parameters(self.natural_parameters)
        return (torch.digamma(alphas) - torch.digamma(alphas.sum())).detach()

    def expected_value(self):
        alphas = self.to_std_parameters(self.natural_parameters)
        return alphas / alphas.sum()

    def log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        alphas = self.to_std_parameters(natural_parameters)
        return torch.lgamma(alphas).sum() - torch.lgamma(alphas.sum())


__all__ = ['DirichletPrior']
