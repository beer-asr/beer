'''This module implements densities member of the Exponential Family of
Distribution.

'''

import torch
import torch.autograd


def log_likelihood(natural_params, log_norm, sufficient_statistics,
                   log_base_measure):
    'L = \eta^T T(X) - A(\eta) + B(X)'
    return sufficient_statistics @ natural_params - log_norm + log_base_measure


def expected_sufficient_statistics(natural_params, log_norm_fn):
    '''E[T(X)] = \nabla_{\eta} A(\eta)

    Note:
        as a side computation, the function also returns the actual
        value of the log-normalization function.

    '''
    natural_params.grad.zero_()
    log_norm = log_norm_fn(natural_params)
    torch.autograd.backward(log_norm)
    return natural_params, natural_params.grad


def bregman_divergence(F_p, F_q, grad_F_q, p, q):
    # (Invalid Argument Name) pylint: disable=C0103
    # In this case, being verbose would be more confusing than anything
    # else. The name of the variables were picked up to match the
    # Wikipedia page about the Bregman divergence.
    '''General implementation of the Bregman divergence:
        F(p) - F(q) - \nabla F(q)^T (p - q)

    Note:
        The Kullback-Leibler divergence between two members of the
        exponential family of the same type (Gaussian-Gaussian,
        Gamma-Gamma, ...) corresponds to a specific type of Bregman
        divergence.

    '''
    return F_p - F_q - grad_F_q @ (p - q)


class ExpFamilyDensity:
    '''General implementation of a member of a Exponential Family of
    Distribution.

    '''

    def __init__(self, natural_params, log_base_measure_fn,
                 sufficient_statistics_fn, log_norm_fn):
        # This will be initialized when setting the natural params
        # property.
        self._log_norm = None
        self._expected_sufficient_statistics = None
        self._natural_params = None

        self._log_base_measure_fn = log_base_measure_fn
        self._s_stats = sufficient_statistics_fn
        self._log_norm_fn = log_norm_fn
        self.natural_params = natural_params

    @property
    def natural_params(self):
        'Natural parameters of the density'
        return self._natural_params

    @natural_params.setter
    def natural_params(self, value):
        self._expected_sufficient_statistics, self._log_norm = \
            expected_sufficient_statistics(value, self._log_norm_fn)
        self._natural_params = value
