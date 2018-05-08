'''This module implements (conjugate) prior densities member of the
Exponential Family of Distribution.

'''

import math
import torch
import torch.autograd as ta


def _bregman_divergence(f_val1, f_val2, grad_f_val2, val1, val2):
    return f_val1 - f_val2 - grad_f_val2 @ (val1 - val2)


def _exp_stats_and_log_norm(natural_params, log_norm_fn, args):
    if natural_params.grad is not None:
        natural_params.grad.zero_()
    log_norm = log_norm_fn(natural_params, **args)
    ta.backward(log_norm)
    return natural_params.grad, log_norm


# The following code compute the log of the determinant of a
# positive definite matrix. This is equivalent to:
#   >>> torch.log(torch.det(mat))
# Note: the hook is necessary to correct the gradient as pytorch
# will return upper triangular gradient.
def _logdet(mat):
    mat.register_hook(lambda grad: .5 * (grad + grad.t()))
    return 2 * torch.log(torch.diag(torch.potrf(mat))).sum()


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


def _jointnormalgamma_split_nparams(natural_params, ncomp):
    # Retrieve the 4 natural parameters organized as
    # follows:
    #   [ np1_1, ..., np1_D, np2_1_1, ..., np2_k_D, np3_1, ..., np3_1_D,
    #     np3_k_D, np4_1, ..., np4_D]
    dim = len(natural_params) // (2 + 2 * ncomp)
    np1 = natural_params[:dim]
    np2s = natural_params[dim: dim + dim * ncomp]
    np3s = natural_params[dim + dim * ncomp: dim + 2 * dim * ncomp]
    np4 = natural_params[dim + 2 * dim * ncomp:]
    return np1, np2s.view(ncomp, dim), np3s.view(ncomp, dim), np4, dim


def _jointnormalgamma_log_norm(natural_params, ncomp):
    np1, np2s, np3s, np4, dim = _jointnormalgamma_split_nparams(natural_params,
                                                                ncomp)
    lognorm = torch.lgamma(.5 * (np4 + 1)).sum()
    lognorm += -.5 * torch.log(np3s).sum()
    tmp = ((np2s ** 2) / np3s).view(ncomp, dim)
    lognorm += torch.sum(-.5 * (np4 + 1) * \
        torch.log(.5 * (np1 - tmp.sum(dim=0))))
    return lognorm


def _normalwishart_split_nparams(natural_params):
    # We need to retrieve the 4 natural parameters organized as
    # follows:
    #   [ np1_1, ..., np1_D^2, np2_1, ..., np2_D, np3, np4]
    #
    # The dimension D is found by solving the polynomial:
    #   D^2 + D - len(self.natural_params[:-2]) = 0
    dim = int(.5 * (-1 + math.sqrt(1 + 4 * len(natural_params[:-2]))))
    np1, np2 = natural_params[:int(dim ** 2)].view(dim, dim), \
         natural_params[int(dim ** 2):-2]
    np3, np4 = natural_params[-2:]
    return np1, np2, np3, np4, dim

def _jointnormalwishart_split_nparams(natural_params, ncomp):
    # We need to retrieve the 4 natural parameters organized as
    # follows:
    #   [ np1_1, ..., np1_D^2, np2_1_1, ..., np2_k_D, np3_1, ...,
    #     np3_k, np4]
    #
    # The dimension D is found by solving the polynomial:
    #   D^2 + ncomp * D - len(self.natural_params[:-(ncomp + 1]) = 0
    dim = int(.5 * (-ncomp + math.sqrt(ncomp**2 + \
        4 * len(natural_params[:-(ncomp + 1)]))))
    np1, np2s = natural_params[:int(dim ** 2)].view(dim, dim), \
         natural_params[int(dim ** 2):-(ncomp + 1)].view(ncomp, dim)
    np3s = natural_params[-(ncomp + 1):-1]
    np4 = natural_params[-1]
    return np1, np2s, np3s, np4, dim


def _normal_fc_split_nparams(natural_params):
    # We need to retrieve the 2 natural parameters organized as
    # follows:
    #   [ np1_1, ..., np1_D^2, np2_1, ..., np2_D]
    #
    # The dimension D is found by solving the polynomial:
    #   D^2 + D - len(self.natural_params) = 0
    dim = int(.5 * (-1 + math.sqrt(1 + 4 * len(natural_params))))
    np1, np2 = natural_params[:int(dim ** 2)].view(dim, dim), \
         natural_params[int(dim ** 2):]
    return np1, np2, dim


def _normal_fc_log_norm(natural_params):
    np1, np2, _ = _normal_fc_split_nparams(natural_params)
    inv_np1 = torch.inverse(np1)
    return -.5 * _logdet(-2 * np1) - .25 * ((np2[None, :] @ inv_np1) @ np2)[0]


def _normalwishart_log_norm(natural_params):
    np1, np2, np3, np4, dim = _normalwishart_split_nparams(natural_params)
    lognorm = .5 * ((np4 + dim) * dim * math.log(2) - dim * torch.log(np3))
    lognorm += -.5 * (np4 + dim) * _logdet(np1 - torch.ger(np2, np2)/np3)
    seq = ta.Variable(torch.arange(1, dim + 1, 1).type(natural_params.type()))
    lognorm += torch.lgamma(.5 * (np4 + dim + 1 - seq)).sum()
    return lognorm


def _jointnormalwishart_log_norm(natural_params, ncomp):
    np1, np2s, np3s, np4, dim = _jointnormalwishart_split_nparams(natural_params,
                                                                  ncomp=ncomp)
    lognorm = .5 * ((np4 + dim) * dim * math.log(2) - dim * torch.log(np3s).sum())
    quad_exp = ((np2s[:, None, :] * np2s[:, :, None]) / \
        np3s[:, None, None]).sum(dim=0)
    lognorm += -.5 * (np4 + dim) * _logdet(np1 - quad_exp)
    seq = ta.Variable(torch.arange(1, dim + 1, 1).type(natural_params.type()))
    lognorm += torch.lgamma(.5 * (np4 + dim + 1 - seq)).sum()
    return lognorm


class ExpFamilyPrior:
    '''General implementation of a member of a Exponential Family of
    Distribution prior.

    '''

    # pylint: disable=W0102
    # Dangerous default value {}.
    def __init__(self, natural_params, log_norm_fn, args={}):
        # This will be initialized when setting the natural params
        # property.
        self._log_norm = None
        self._expected_sufficient_statistics = None
        self._natural_params = None
        self._args = args

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
            _exp_stats_and_log_norm(value, self._log_norm_fn, self._args)
        self._natural_params = value


def kl_div(model1, model2):
    '''Kullback-Leibler divergence between two densities of the same
    type.

    '''
    return _bregman_divergence(model2.log_norm, model1.log_norm,
                               model1.expected_sufficient_statistics,
                               model2.natural_params, model1.natural_params)


# pylint: disable=C0103
# Invalid function name.
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


# pylint: disable=C0103
# Invalid function name.
def NormalGammaPrior(mean, precision, prior_counts):
    '''Create a NormalGamma density function.

    Args:
        mean (Tensor): Mean of the Normal.
        precision (Tensor): Mean of the Gamma.
        prior_counts (float): Strength of the prior.

    Returns:
        A NormalGamma density.

    '''
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

# pylint: disable=C0103
# Invalid function name.
def JointNormalGammaPrior(means, prec, prior_counts):
    '''Create a joint Normal-Gamma density function.

    Note:
        By "joint" normal-gamma density we mean a set of independent
        Normal density sharing the same diagonal covariance matrix
        (up to a multiplicative constant) and the probability over the
        diagonal of the covariance matrix is given by a D indenpendent
        Gamma distributions.

    Args:
        means (Tensor): Expected mean of the Normal densities.
        prec (Tensor): Expected precision for each dimension.
        prior_counts (float): Strength of the prior.

    Returns:
        ``JointNormalGammaPrior``

    '''
    dim = means.size(1)
    ncomp = len(means)
    natural_params = ta.Variable(torch.cat([
        (prior_counts * (means**2).sum(dim=0) + 2 * prior_counts).view(-1),
        (prior_counts * means).view(-1),
        (torch.ones(ncomp, dim) * prior_counts).type(means.type()).view(-1),
        2 * prec * prior_counts - 1
    ]), requires_grad=True)
    return ExpFamilyPrior(natural_params, _jointnormalgamma_log_norm,
                          args={'ncomp': means.size(0)})


# pylint: disable=C0103
# Invalid function name.
def NormalWishartPrior(mean, cov, prior_counts):
    '''Create a NormalWishart density function.

    Args:
        mean (Tensor): Expected mean of the Normal.
        cov (Tensor): Expected covariance matrix.
        prior_counts (float): Strength of the prior.

    Returns:
        A NormalWishart density.

    '''
    if len(cov.size()) != 2:
        raise ValueError('Expect a (D x D) matrix')

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


# pylint: disable=C0103
# Invalid function name.
def JointNormalWishartPrior(means, cov, prior_counts):
    '''Create a JointNormalWishart density function.

    Note:
        By "joint" normal-wishart density we mean a set of independent
        Normal density sharing the same covariance matrix (up to a
        multiplicative constant) and the probability over the
        covariance matrix is given by a Wishart distribution.

    Args:
        means (Tensor): Expected mean of the Normal densities.
        cov (Tensor): Expected covariance matrix.
        prior_counts (float): Strength of the prior.

    Returns:
        ``JointNormalWishartPrior``

    '''
    if len(cov.size()) != 2:
        raise ValueError('Expect a (D x D) matrix')

    D = means.size(1)
    dof = prior_counts + D
    V = dof * cov
    mmT = (means[:, None, :] * means[:, :, None]).sum(dim=0)
    natural_params = ta.Variable(torch.cat([
        (prior_counts * mmT + V).view(-1),
        prior_counts * means.view(-1),
        (torch.ones(means.size(0)) * prior_counts).type(means.type()),
        (torch.ones(1) * (dof - (D))).type(means.type())
    ]), requires_grad=True)
    return ExpFamilyPrior(natural_params, _jointnormalwishart_log_norm,
                          args={'ncomp': means.size(0)})


# pylint: disable=C0103
# Invalid function name.
def NormalPrior(mean, cov):
    '''Create a Normal density prior.

    Args:
        mean (Tensor): Expected mean.
        cov (Tensor): Expected covariance of the mean.

    Returns:
        ``NormalPrior``: A Normal density.

    '''
    if len(cov.size()) != 2:
        raise ValueError('Expect a (D x D) matrix')

    natural_params = ta.Variable(torch.cat([
        -.5 * cov.view(-1),
        cov @ mean,
    ]), requires_grad=True)
    return ExpFamilyPrior(natural_params, _normal_fc_log_norm)
