'''This module implements densities member of the Exponential Family of
Distribution.

'''

import torch
import torch.autograd as ta


def _bregman_divergence(F_p, F_q, grad_F_q, p, q):
    # (Invalid Argument Name) pylint: disable=C0103
    return F_p - F_q - grad_F_q @ (p - q)


def _exp_stats_and_log_norm(natural_params, log_norm_fn):
    if natural_params.grad is not None:
        natural_params.grad.zero_()
    log_norm = log_norm_fn(natural_params)
    ta.backward(log_norm)
    return natural_params.grad, log_norm


def _log_likelihood(natural_params, log_norm, sufficient_statistics,
                    log_base_measure):
    # (Anomalous backslash in string) pylint: disable=W1401
    return sufficient_statistics @ natural_params - log_norm + log_base_measure

########################################################################
## Dirichlet distribution.
########################################################################

def _dirichlet_sufficient_statistics(X):
    # (Invalid Argument Name) pylint: disable=C0103
    # (Module 'torch' has no 'log' member) pylint: disable=E1101
    return torch.log(X)


def _dirichlet_log_base_measure(X):
    # (Invalid Argument Name) pylint: disable=C0103
    # (Unused argument 'X') pylint: disable=W0613
    return 0.


def _dirichlet_log_norm(natural_params):
    # (Module 'torch' has no 'lgamma' member) pylint: disable=E1101
    return - torch.lgamma((natural_params + 1).sum()) \
        + torch.lgamma(natural_params + 1).sum()


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

        self._log_norm_fn = log_norm_fn
        self._log_bmeasure_fn = log_base_measure_fn
        self._s_stats_fn = sufficient_statistics_fn
        self.natural_params = natural_params

    @property
    def expected_sufficient_statistics(self):
        'Expected value of the sufficient statistics.'
        return self._expected_sufficient_statistics

    @property
    def log_norm(self):
        'Value of the log-partition function for the given parameters.'
        return self._log_norm

    @property
    def natural_params(self):
        'Natural parameters of the density'
        return self._natural_params

    @natural_params.setter
    def natural_params(self, value):
        self._expected_sufficient_statistics, self._log_norm = \
            _exp_stats_and_log_norm(value, self._log_norm_fn)
        self._natural_params = value

    def log_base_measure(self, X):
        # (Invalid Argument Name) pylint: disable=C0103
        'B(X)'
        return self._log_bmeasure_fn(X)

    def log_likelihood(self, X):
        # (Invalid Argument Name) pylint: disable=C0103
        'Log-likelihood of the data given the parameters of the model.'
        s_stats = self._s_stats_fn(X)
        log_bmeasure = self._log_bmeasure_fn(X)
        return _log_likelihood(self.natural_params, self.log_norm, s_stats,
                               log_bmeasure)

    def sufficient_statistics(self, X):
        # (Invalid Argument Name) pylint: disable=C0103
        'T(X)'
        return self._s_stats_fn(X)


def kl_divergence(model1, model2):
    '''Kullback-Leibler divergence between two densities of the same
    type.

    '''
    return _bregman_divergence(model2.log_norm, model1.log_norm,
                               model1.expected_sufficient_statistics,
                               model2.natural_params, model1.natural_params)


def dirichlet(prior_counts):
    '''Create a Dirichlet density function.

    Args:
        prior_counts (Tensor/Variable): Prior counts for each category.
            If a ``Variable`` is passed, it should be created with
            ``requires_grad=True``.

    Returns:
        A Dirichlet density.

    '''
    natural_params = prior_counts - 1
    if not isinstance(natural_params, ta.Variable):
        natural_params = ta.Variable(natural_params, requires_grad=True)
    return ExpFamilyDensity(
        natural_params,
        _dirichlet_log_base_measure,
        _dirichlet_sufficient_statistics,
        _dirichlet_log_norm
    )
