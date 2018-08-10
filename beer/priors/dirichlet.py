'''Implementation of the Dirichlet prior.'''

import torch
from .baseprior import ExpFamilyPrior


class DirichletPrior(ExpFamilyPrior):
    '''Dirichlet Distribution.

    parameters:
        alpha[k] > 0

    natural parameters:
        eta[k] = alphas[k] - 1

    '''

    @property
    def strength(self):
        alphas = self.to_std_parameters(self.natural_params)
        mean = alphas / alphas.sum()
        return alphas[0] / mean[0]

    @strength.setter
    def strength(self, value):
        pass

    def to_std_parameters(self, natural_params):
        return natural_params + 1

    def to_natural_parameters(self, std_params):
        return std_params - 1

    def expected_sufficient_statistics(self):
        alphas = self.to_std_parameters(self.natural_params)
        return torch.digamma(alphas) - torch.digamma(alphas.sum())

    def expected_value(self):
        alphas = self.to_std_parameters(self.natural_params)
        return alphas / alphas.sum()

    def log_norm(self, natural_hparams):
        alphas = self.to_std_parameters(self.natural_params)
        return torch.lgamma(alphas).sum() - torch.lgamma(alphas.sum())


__all__ = ['DirichletPrior']
