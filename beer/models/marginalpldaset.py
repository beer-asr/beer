'''Bayesian Subspace models.'''


import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianModelSet
from .ppca import kl_div_std_norm
from .normal import NormalFullCovariance
from .normalset import NormalSetElement
from ..expfamilyprior import NormalWishartPrior
from ..expfamilyprior import NormalIsotropicCovariancePrior
from ..expfamilyprior import NormalFullCovariancePrior
from ..expfamilyprior import WishartPrior
from ..utils import make_symposdef


class MarginalPLDASet(BayesianModelSet):
    '''PLDA set where the noise/class subspace are marginalized.'''

    def __init__(self, normal, prior_means, posterior_means, prior_class_cov,
                 posterior_class_cov):
        super().__init__()
        self.normal = normal
        self.class_mean_params = BayesianParameterSet([
            BayesianParameter(prior, posterior)
            for prior, posterior in zip(prior_means, posterior_means)
        ])
        self.class_cov_param = BayesianParameter(prior_class_cov,
                                                 posterior_class_cov)
        self.class_cov_param.register_callback(self.on_class_cov_update)

    def on_class_cov_update(self):
        cov = make_symposdef(self.class_cov)
        p_mean = torch.zeros_like(self.mean)
        prior = NormalFullCovariancePrior(p_mean, cov)
        for param in self.class_mean_params:
            param.prior = prior

    @property
    def mean(self):
        return self.normal.mean

    @property
    def cov(self):
        return self.normal.cov

    @property
    def class_means(self):
        means = []
        for mean_param in self.class_mean_params:
            _, mean = mean_param.expected_value(concatenated=False)
            means.append(mean)
        return torch.stack(means)

    @property
    def class_covs(self):
        covs = []
        for mean_param in self.class_mean_params:
            quad, mean = mean_param.expected_value(concatenated=False)
            covs.append(quad - torch.ger(mean, mean))
        return torch.stack(covs)

    @property
    def class_cov(self):
        np1, np2 = self.class_cov_param.expected_value(concatenated=False)
        return torch.inverse(np1)

    def reset_class_means(self, n_classes, noise_std=1, prior_strength=1.):
        '''Sample a new set of class means while keeping the prior
        over the the class mean covariance matrix..

        Args:
            n_classes (int): new number of components.
            noise_std (float): Standard deviation of the noise for the
                random initialization.
            prior_strength (float): Strength of the class means' prior.

        '''
        mean = self.mean
        dim = len(mean)
        dtype, device = mean.dtype, mean.device
        class_cov = make_symposdef(self.class_cov)
        cov = self[0].cov / prior_strength
        lower_cholesky = torch.potrf(class_cov, upper=False)

        # Set the prior of the class covariance matrix to the current
        # posterior.
        new_prior = self.class_cov_param.posterior.copy_with_new_params(
            self.class_cov_param.posterior.natural_hparams
        )
        self.class_cov_param.prior = new_prior

        # Random initialization of the new class means.
        noise = torch.randn(n_classes, dim, dtype=dtype, device=device)
        init_c_means = (noise_std * noise @ lower_cholesky.t())

        # Create the new priors/posteriors
        class_mean_prior = NormalFullCovariancePrior(torch.zeros_like(mean), cov)
        class_mean_priors = [class_mean_prior] * n_classes
        class_mean_posteriors = []
        for post_mean in init_c_means:
            class_mean_posteriors.append(NormalFullCovariancePrior(post_mean, cov))

        # Set the new parameters
        self.class_mean_params = BayesianParameterSet([
            BayesianParameter(prior, posterior)
            for prior, posterior in zip(class_mean_priors, class_mean_posteriors)
        ])

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return [
            *self.normal.mean_field_factorization(),
            [*self.class_mean_params],
            [self.class_cov_param]
        ]

    def sufficient_statistics(self, data):
        # We need the raw data to accumualte the s. statistics.
        self.cache['data'] = data

        means = self.class_means
        covs = self.class_covs
        data_means = data[:, None, :] - means[None]
        data_mean_quad = data_means[:, :, :, None] * data_means[:, :, None, :]
        return torch.cat([
            (data_mean_quad + covs[None]).view(len(data), len(self), -1),
            data_means,
            torch.ones(len(data), len(self), 2, dtype=data.dtype,
                       device=data.device),
        ], dim=-1)

    def forward(self, s_stats):
        feadim = len(self.normal.mean)
        nparams = self.normal.mean_precision.expected_value()
        exp_llh = s_stats @ nparams
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        resps = parent_msg
        dtype = resps.dtype
        device = resps.device
        data = self.cache['data']
        np1, _, _, _ = self.normal.mean_precision.expected_value(concatenated=False)
        prec = make_symposdef(-2 * np1)
        prec_data_mean = (data - self.normal.mean) @ prec

        acc_stats = self.normal.accumulate(
            (s_stats * resps[:, :, None]).sum(dim=1))

        # Accumulate the statistics for the class means.
        acc_resps = resps.sum(dim=0)
        acc_prec_data_mean = (prec_data_mean[:, :, None] @ resps[:, None, :]).sum(dim=0).t()
        for i, mean_param in enumerate(self.class_mean_params):
            class_mean_acc_stats = {
                mean_param: torch.cat([
                    (-.5 * acc_resps[i] * prec).view(-1),
                    acc_prec_data_mean[i]
                ])
            }
            acc_stats.update(class_mean_acc_stats)

        # Accumulate statistics for the class cov (i.e. the between
        # class covariance matrix).
        _, prior_mean = self.class_mean_params[0].prior.split_sufficient_statistics(
            self.class_mean_params[0].prior.expected_sufficient_statistics
        )
        class_means_quad, class_mean_mean = [], []
        for mean_param in self.class_mean_params:
            quad, mean = mean_param.expected_value(concatenated=False)
            class_means_quad.append(quad)
            mean_prior_mean = torch.ger(mean, prior_mean)
            class_mean_mean.append(mean_prior_mean + mean_prior_mean.t())
        class_means_quad = torch.stack(class_means_quad)
        class_means_mean = torch.stack(class_mean_mean)
        prior_mean_mean = torch.ger(prior_mean, prior_mean)
        stats = class_means_quad - class_means_mean + prior_mean_mean
        stats = make_symposdef(stats.sum(dim=0))
        class_cov_acc_stats = {
            self.class_cov_param: torch.cat([
                -.5 * stats.view(-1),
                .5 * torch.tensor(len(self), dtype=dtype, device=device).view(1)
            ])
        }
        acc_stats.update(class_cov_acc_stats)

        return acc_stats

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        mean = self.normal.mean + self.class_means[key]
        cov = self.normal.cov
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return len(self.class_mean_params)


def _init_params_random(global_mean, noise_std, class_means, m_noise_subspace,
                        m_class_subspace, dtype, device):
    noise = torch.randn(*m_noise_subspace.size(), dtype=dtype, device=device)
    init_noise_s = m_noise_subspace + noise_std * noise

    noise = torch.randn(*m_class_subspace.size(), dtype=dtype, device=device)
    init_class_s = m_class_subspace + noise_std * noise

    means = []
    for i in range(class_means):
        r_mean = noise_std * torch.randn(m_class_subspace.shape[0],
                                         dtype=dtype, device=device)
        means.append(r_mean)

    return global_mean, init_noise_s, init_class_s, means


def create(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    n_classes = model_conf['size']
    noise_std = model_conf['noise_std']
    prior_strength = model_conf['prior_strength']

    # Normal distribution.
    normal =  NormalFullCovariance.create(mean, variance, prior_strength,
                                          noise_std)

    # Class means
    class_means = torch.zeros(n_classes, len(mean), dtype=dtype,device=device)
    init_class_means = []
    for i in range(n_classes):
        r_mean = noise_std * torch.randn(len(mean), dtype=dtype, device=device)
        init_class_means.append(r_mean)

    cov = torch.eye(len(mean), dtype=dtype, device=device) / prior_strength
    class_mean_priors, class_mean_posteriors = [], []
    class_mean_prior = NormalFullCovariancePrior(torch.zeros_like(mean), cov)
    for post_mean in init_class_means:
        class_mean_priors.append(class_mean_prior)
        class_mean_posteriors.append(NormalFullCovariancePrior(post_mean, cov))

    class_cov = torch.eye(len(mean), dtype=dtype, device=device)
    dof = torch.tensor(prior_strength + len(mean) - 1, dtype=dtype, device=device)
    class_cov_prior = WishartPrior(class_cov, dof)
    class_cov_posterior = WishartPrior(class_cov, dof)

    return MarginalPLDASet(
        normal,
        class_mean_priors,
        class_mean_posteriors,
        class_cov_prior,
        class_cov_posterior
    )


__all__ = ['MarginalPLDASet']
