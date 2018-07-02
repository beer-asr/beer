'''Bayesian Subspace models.'''


import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel
from ..expfamilyprior import GammaPrior
from ..expfamilyprior import NormalIsotropicCovariancePrior
from ..expfamilyprior import MatrixNormalPrior


def kl_div_std_norm(means, cov):
    '''KL divergence between a set of Normal distributions with a
    shared covariance matrix and a standard Normal N(0, I).

    Args:
        means (``torch.Tensor[N, dim]``): Means of the
            Normal distributions where N is  the number of frames and
            dim is the dimension of the random variable.
        cov (``torch.Tensor[s_dim, s_dim]``): Shared covariance matrix.

    Returns:
        ``torch.Tensor[N]``: Per-distribution KL-divergence.

    '''
    dim = means.size(1)
    _, logdet = torch.slogdet(cov)
    return .5 * (-dim - logdet + torch.trace(cov) + \
        torch.sum(means ** 2, dim=1))



########################################################################
# Probabilistic Principal Component Analysis (PPCA)
########################################################################

class PPCA(BayesianModel):
    '''Probabilistic Principal Component Analysis (PPCA).

    Attributes:
        mean (``torch.Tensor``): Expected mean.
        precision (``torch.Tensor[1]``): Expected precision (scalar).
        subspace (``torch.Tensor[subspace_dim, data_dim]``): Mean of the
            matrix definining the prior.

    Example:
        >>> subspace_dim, dim = 2, 4
        >>> mean = torch.zeros(dim)
        >>> precision = 1.
        >>> subspace = torch.randn(subspace_dim ,dim)
        >>> ppca = beer.PPCA.create(mean, precision, subspace)
        >>> ppca.mean
        tensor([ 0.,  0.,  0.,  0.])
        >>> ppca.precision
        tensor(1.)
        >>> ppca.subspace
        tensor([[ 0.0328, -2.8251,  2.6031, -0.9721],
                [ 0.5193,  0.6301, -0.3425,  1.2708]])

    '''

    def __init__(self, prior_mean, posterior_mean, prior_prec, posterior_prec,
                 prior_subspace, posterior_subspace):
        '''
        Args:
            prior_mean (``NormalIsotropicCovariancePrior``): Prior over
                the global mean.
            posterior_mean (``NormalIsotropicCovariancePrior``):
                Posterior over the global mean.
            prior_prec (``GammaPrior``): Prior over the global precision.
            posterior_prec (``GammaPrior``): Posterior over the global
                precision.
            prior_subspace (``MatrixNormalPrior``): Prior over
                the subpace (i.e. the linear transform of the model).
            posterior_subspace (``MatrixNormalPrior``): Posterior over
                the subspace (i.e. the linear transform of the model).
            subspace_dim (int): Dimension of the subspace.

        '''
        super().__init__()
        self.mean_param = BayesianParameter(prior_mean, posterior_mean)
        self.precision_param = BayesianParameter(prior_prec, posterior_prec)
        self.subspace_param = BayesianParameter(prior_subspace,
                                                posterior_subspace)
        self._subspace_dim, self._data_dim = self.subspace.size()

    @property
    def mean(self):
        _, mean = self.mean_param.expected_value(concatenated=False)
        return mean

    @property
    def precision(self):
        return self.precision_param.expected_value()[1]

    @property
    def subspace(self):
        _, exp_value2 = self.subspace_param.expected_value(concatenated=False)
        return exp_value2

    def _get_expectation(self):
        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        s_quad, s_mean = self.subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)
        return log_prec, prec, s_quad, s_mean, m_quad, m_mean

    def _compute_distance_term(self, s_stats, l_means, l_quad):
        _, _, s_quad, s_mean, m_quad, m_mean = self._get_expectation()
        data_mean = s_stats[:, 1:] - m_mean.view(1, -1)
        distance_term = torch.zeros(len(s_stats), dtype=s_stats.dtype,
                                    device=s_stats.device)
        distance_term += s_stats[:, 0]
        distance_term += - 2 * s_stats[:, 1:] @ m_mean
        distance_term += - 2 * torch.sum((l_means @ s_mean) * data_mean, dim=1)
        distance_term += l_quad.view(len(s_stats), -1) @ s_quad.view(-1)
        distance_term += m_quad
        return distance_term

    def latent_posterior(self, stats):
        data = stats[:, 1:]
        _, prec, s_quad, s_mean, _, m_mean = self._get_expectation()
        lposterior_cov = torch.inverse(
            torch.eye(self._subspace_dim, dtype=stats.dtype,
                      device=stats.device) + prec * s_quad)
        lposterior_means = prec * lposterior_cov @ s_mean @ (data - m_mean).t()
        return lposterior_means.t(), lposterior_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @property
    def grouped_parameters(self):
        return [
            [self.mean_param, self.precision_param],
            [self.subspace_param]
        ]

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data],
                         dim=-1)

    def float(self):
        return self.__class__(
            self.mean_param.prior.float(),
            self.mean_param.posterior.float(),
            self.precision_param.prior.float(),
            self.precision_param.posterior.float(),
            self.subspace_param.prior.float(),
            self.subspace_param.posterior.float()
        )

    def double(self):
        return self.__class__(
            self.mean_param.prior.double(),
            self.mean_param.posterior.double(),
            self.precision_param.prior.double(),
            self.precision_param.posterior.double(),
            self.subspace_param.prior.double(),
            self.subspace_param.posterior.double()
        )

    def to(self, device):
        return self.__class__(
            self.mean_param.prior.to(device),
            self.mean_param.posterior.to(device),
            self.precision_param.prior.to(device),
            self.precision_param.posterior.to(device),
            self.subspace_param.prior.to(device),
            self.subspace_param.posterior.to(device)
        )


    def forward(self, s_stats):
        feadim = s_stats.size(1) - 1
        l_means, l_cov = self.latent_posterior(s_stats)
        l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
        l_kl_div = kl_div_std_norm(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, _, s_mean, _, m_mean = self._get_expectation()

        distance_term = self._compute_distance_term(s_stats, l_means, l_quad)
        exp_llh = torch.zeros(len(s_stats), dtype=s_stats.dtype,
                              device=s_stats.device)
        exp_llh += -.5 * feadim * math.log(2 * math.pi)
        exp_llh += .5 * feadim * log_prec
        exp_llh += -.5 * prec * distance_term

        # Cache some computation for a quick accumulation of the
        # sufficient statistics.
        self.cache['distance'] = distance_term.sum()
        self.cache['latent_means'] = l_means
        self.cache['latent_quad'] = l_quad
        self.cache['kl_divergence'] = l_kl_div
        self.cache['precision'] = prec
        self.cache['subspace_mean'] = s_mean
        self.cache['mean_mean'] = m_mean

        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        # Load cached values and clear the cache.
        distance_term = self.cache['distance']
        l_means = self.cache['latent_means']
        l_quad = self.cache['latent_quad']
        prec = self.cache['precision']
        s_mean = self.cache['subspace_mean']
        m_mean = self.cache['mean_mean']
        self.clear_cache()

        dtype = s_stats.dtype
        device = s_stats.device
        feadim = s_stats.size(1) - 1

        data_mean = s_stats[:, 1:] - m_mean[None, :]
        acc_s_mean = (l_means.t() @ data_mean)
        return {
            self.precision_param: torch.cat([
                .5 * torch.tensor(len(s_stats) * feadim, dtype=dtype,
                                  device=device).view(1),
                -.5 * distance_term.view(1)
            ]),
            self.mean_param: torch.cat([
                - .5 * torch.tensor(len(s_stats) * prec, dtype=dtype,
                                    device=device).view(1),
                prec * torch.sum(s_stats[:, 1:] - l_means @ s_mean, dim=0)
            ]),
            self.subspace_param:  torch.cat([
                - .5 * prec * l_quad.sum(dim=0).view(-1),
                prec * acc_s_mean.view(-1)
            ])
        }

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([torch.sum(mean ** 2 + var, dim=1).view(-1, 1), mean],
                         dim=1)

    def local_kl_div_posterior_prior(self, parent_msg=None):
        return self.cache['kl_divergence']

    def expected_natural_params(self, mean, var, latent_variables=None,
                                nsamples=1):
        '''Interface for the VAE model. Returns the expected value of the
        natural params of the latent model given the per-frame means
        and variances.

        Args:
            mean (Tensor): Per-frame mean of the posterior distribution.
            var (Tensor): Per-frame variance of the posterior
                distribution.
            labels (Tensor): Frame labelling (if any).
            nsamples (int): Number of samples to estimate the
                natural parameters (ignored).

        Returns:
            (Tensor): Expected value of the natural parameters.

        Note:
            The expected natural parameters can be estimated in close
            form and therefore does not requires sampling. Hence,
            the parameters `nsamples` will be ignored.

        '''
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)

        if latent_variables is not None:
            l_means = latent_variables
            l_quad = l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = torch.zeros(len(s_stats), dtype=mean.dtype,
                                   device=mean.device)
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = kl_div_std_norm(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, s_quad, s_mean, m_quad, m_mean = self._get_expectation()

        np1 = -.5 * prec * torch.ones(len(s_stats), mean.size(1),
                                      dtype=mean.dtype, device=mean.device)
        np2 = prec * (l_means @ s_mean + m_mean)
        np3 = torch.zeros(len(s_stats), mean.size(1), dtype=mean.dtype,
                          device=mean.device)
        np3 += -.5 * prec * (l_quad.view(len(s_stats), -1) @ s_quad.view(-1)).view(-1, 1)
        np3 += -(prec * l_means @ s_mean @ m_mean).reshape(-1, 1)
        np3 += -.5 * prec * m_quad
        np3 /= self._data_dim
        np4 = .5 * log_prec * torch.ones(s_stats.size(0), mean.size(1),
                                         dtype=mean.dtype, device=mean.device)

        # Cache some computation for a quick accumulation of the
        # sufficient statistics.
        self.cache['distance'] = self._compute_distance_term(s_stats, l_means,
                                                             l_quad).sum()
        self.cache['latent_means'] = l_means
        self.cache['latent_quad'] = l_quad
        self.cache['kl_divergence'] = l_kl_div
        self.cache['precision'] = prec
        self.cache['subspace_mean'] = s_mean
        self.cache['mean_mean'] = m_mean
        return torch.cat([np1, np2, np3, np4], dim=1), s_stats


def create(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    dim_subspace = model_conf['dim_subspace']
    noise_std = model_conf['noise_std']
    prior_strength = model_conf['prior_strength']

    # Precision.
    shape = torch.tensor([prior_strength], dtype=dtype, device=device)
    rate = torch.tensor([prior_strength * variance.sum()], dtype=dtype,
                        device=device)
    prior_prec = GammaPrior(shape, rate)
    posterior_prec = GammaPrior(shape, rate)
    mean_variance = torch.tensor([1. / float(prior_strength)], dtype=dtype,
                                 device=device)

    # Mean.
    prior_mean = NormalIsotropicCovariancePrior(mean, mean_variance)
    posterior_mean = NormalIsotropicCovariancePrior(mean, mean_variance)

    # Subspace.
    mean_subspace = torch.zeros(dim_subspace, len(mean), dtype=dtype,
                                device=device)
    cov = torch.eye(dim_subspace, dtype=dtype, device=device) / prior_strength
    prior_subspace = MatrixNormalPrior(mean_subspace, cov)
    rand_init = mean_subspace + noise_std * torch.randn(*mean_subspace.size(),
                                                        dtype=dtype,
                                                        device=device)
    posterior_subspace = MatrixNormalPrior(rand_init, cov)

    return PPCA(prior_mean, posterior_mean, prior_prec, posterior_prec,
                prior_subspace, posterior_subspace)


__all__ = ['PPCA']
