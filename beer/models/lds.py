
from functools import lru_cache
import numpy as np
import torch

from .parameters import ConstantParameter
from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModelSet
from .linearreg import LinearRegressionSet


def compute_dct_bases(ndim, n_dct_coeff, dtype, device):
    dct_bases = np.zeros((ndim, n_dct_coeff))
    for k in range(n_dct_coeff):
        dct_bases[:, k] = np.cos((np.pi / ndim) * (np.arange(ndim) + 0.5 ) * k)
    return torch.from_numpy(dct_bases).type(dtype).to(device)



class LDSSet(BayesianModelSet):
    '''Set of Bayesian Linear Dynamical System (LDS).

    Attributes:
        filters: Filtering matrix parameter.
        precision: precision parameter.

    '''


    @classmethod
    def create(cls, mean, variance, size, memory, n_dct_bases=5, prior_strength=1.,
               noise_std=.1, weights_variance=.01):
        mean_weights = torch.zeros(n_dct_bases * len(mean), len(mean))
        lr_set = LinearRegressionSet.create(size, mean_weights, variance,
                                             prior_strength, weights_variance,
                                             noise_std)
        dct_bases = compute_dct_bases(memory, n_dct_bases, mean.dtype,
                                      mean.device)
        return cls(lr_set, memory, dct_bases)


    def __init__(self, lr_set, memory, dct_bases):
        super().__init__()
        self.lr_set = lr_set
        self.memory = memory
        self.dct_bases = ConstantParameter(dct_bases.detach())

    def __len__(self):
        return len(self.lr_set)

    def __getitem__(self, key):
        return self.lr_set[key]

    def mean_field_factorization(self):
        return self.lr_set.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.lr_set.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        X = stats[:, 1:-2]

        # Get the context .
        pX = torch.nn.functional.pad(X, pad=(0, 0, self.memory, 0)).detach()
        stacked_X = pX[:-1].unfold(0, self.memory, 1)

        # Compute the regressors.
        phi = (stacked_X @ self.dct_bases.value).reshape(len(X), -1)

        return self.lr_set.expected_log_likelihood(stats, regressors=phi)

    def accumulate(self, stats, resps):
        return self.lr_set.accumulate(stats, resps)


__all__ = [
    'LDSSet',
]
