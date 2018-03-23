'''This module implements (conjugate) prior densities member of the
Exponential Family of Distribution.

'''

import math
import torch
import torch.autograd as ta


def _bregman_divergence(F_p, F_q, grad_F_q, p, q):
    return F_p - F_q - grad_F_q @ (p - q)


def _exp_stats_and_log_norm(natural_params, log_norm_fn):
    if natural_params.grad is not None:
        natural_params.grad.zero_()
    log_norm = log_norm_fn(natural_params)
    ta.backward(log_norm)
    return natural_params.grad, log_norm


########################################################################
## Densities log-normalizer functions.
########################################################################

def _dirichlet_log_norm(natural_params):
    return - torch.lgamma((natural_params + 1).sum()) \
        + torch.lgamma(natural_params + 1).sum()


def _normalgamma_log_norm(natural_params):
    np1, np2, np3, np4 = natural_params.view(4, -1)
    lognorm = torch.lgamma(.5 * (np4 + 1))
    lognorm += -.5 * torch.log(np3)
    lognorm += -.5 * (np4 + 1) * torch.log(.5 * (np1 - ((np2**2) / np3)))
    return torch.sum(lognorm)


def _normalwishart_split_nparams(natural_params):
    # We need to retrieve the 4 natural parameters organized as
    # follows:
    #   [ np1_1, ..., np1_D^2, np2_1, ..., np2_D, np3, np4]
    #
    # The dimension D is found by solving the polynomial:
    #   D^2 + D - len(self.natural_params[:-2]) = 0
    D = int(.5 * (-1 + math.sqrt(1 + 4 * len(natural_params[:-2]))))
    np1, np2 = natural_params[:int(D**2)].view(D, D), \
         natural_params[int(D**2):-2]
    np3, np4 = natural_params[-2:]
    return np1, np2, np3, np4, D


def _normalwishart_log_norm(natural_params):
    np1, np2, np3, np4, D = _normalwishart_split_nparams(natural_params)
    lognorm = .5 * ((np4 + D) * D * math.log(2) - D * torch.log(np3))
    tmp = np1 - torch.ger(np2, np2) /np3

    # The following code compute the log of the determinant of a
    # positive definite matrix. This is equivalent to:
    #   >>> torch.log(torch.det(tmp))
    # Note: the hook is necessary to correct the gradient as pytorch
    # will return upper triangular gradient.
    tmp.register_hook(lambda grad: .5 * (grad + grad.t()))
    logdet = 2 * torch.log(torch.diag(torch.potrf(tmp))).sum()

    lognorm += -.5 * (np4 + D) * logdet
    seq = ta.Variable(torch.arange(1, D + 1, 1).type(natural_params.type()))
    lognorm += torch.lgamma(.5 * (np4 + D + 1 - seq)).sum()
    return lognorm


class ExpFamilyPrior:
    '''General implementation of a member of a Exponential Family of
    Distribution prior.

    '''

    def __init__(self, natural_params, log_norm_fn):
        # This will be initialized when setting the natural params
        # property.
        self._log_norm = None
        self._expected_sufficient_statistics = None
        self._natural_params = None

        self._log_norm_fn = log_norm_fn
        self.natural_params = natural_params

    @property
    def expected_sufficient_statistics(self):
        'Expected value of the sufficient statistics.'
        return self._expected_sufficient_statistics.data

    @property
    def log_norm(self):
        'Value of the log-partition function for the given parameters.'
        return self._log_norm.data

    @property
    def natural_params(self):
        'Natural parameters of the density'
        return self._natural_params.data

    @natural_params.setter
    def natural_params(self, value):
        self._expected_sufficient_statistics, self._log_norm = \
            _exp_stats_and_log_norm(value, self._log_norm_fn)
        self._natural_params = value


def kl_div(model1, model2):
    '''Kullback-Leibler divergence between two densities of the same
    type.

    '''
    return _bregman_divergence(model2.log_norm, model1.log_norm,
                               model1.expected_sufficient_statistics,
                               model2.natural_params, model1.natural_params)


def DirichletPrior(prior_counts):
    '''Create a Dirichlet density function.

    Args:
        prior_counts (Tensor): Prior counts for each category.

    Returns:
        A Dirichlet density.

    '''
    natural_params = prior_counts - 1
    natural_params = ta.Variable(natural_params, requires_grad=True)
    return ExpFamilyPrior(natural_params, _dirichlet_log_norm)


def NormalGammaPrior(mean, precision, prior_counts):
    '''Create a NormalGamma density function.

    Args:
        mean (Tensor): Mean of the Normal.
        precision (Tensor): Mean of the Gamma.
        prior_counts (float): Strength of the prior.

    Returns:
        A NormalGamma density.

    '''
    dim = mean.size(0)
    n_mean = mean
    n_precision = prior_counts * torch.ones_like(n_mean)
    g_shapes = precision * prior_counts
    g_rates = prior_counts
    natural_params = ta.Variable(torch.cat([
        n_precision * (n_mean ** 2) + 2 * g_rates,
        n_precision * n_mean,
        n_precision,
        2 * g_shapes - 1
    ]), requires_grad=True)
    return ExpFamilyPrior(natural_params, _normalgamma_log_norm)


def NormalWishartPrior(mean, cov, prior_counts):
    '''Create a NormalWishart density function.

    Args:
        mean (Tensor): Expected mean of the Normal.
        cov (Tensor): Expected covariance matrix.
        prior_counts (float): Strength of the prior.

    Returns:
        A NormalWishart density.

    '''
    if len(cov.size()) != 2: raise ValueError('Expect a (D x D) matrix')

    D = mean.size(0)
    dof = prior_counts + D
    V = dof * cov
    natural_params = ta.Variable(torch.cat([
        (prior_counts * torch.ger(mean, mean) + V).view(-1),
        prior_counts * mean,
        (torch.ones(1) * prior_counts).type(mean.type()),
        (torch.ones(1) * (dof - D)).type(mean.type())
    ]), requires_grad=True)
    return ExpFamilyPrior(natural_params, _normalwishart_log_norm)

