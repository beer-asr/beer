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
from ..utils import make_symposdef


class MarginalPLDASet(BayesianModelSet):
    '''PLDA set where the noise/class subspace are marginalized.'''

    def __init__(self, normal, prior_means, posterior_means):
        super().__init__()
        self.normal = normal
        self.class_mean_params = BayesianParameterSet([
            BayesianParameter(prior, posterior)
            for prior, posterior in zip(prior_means, posterior_means)
        ])

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

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.normal.mean_field_factorization() + \
            [list(self.class_mean_params)]

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
        prec = -2 * np1
        prec_data_mean = (data - self.normal.mean) @ prec

        acc_stats = self.normal.accumulate(
            (s_stats * resps[:, :, None]).sum(dim=1))

        # Accumulate the statistics for the class means.
        #class_mean_acc_stats1 = class_s_quad
        #class_mean_acc_stats1 = make_symposdef(class_mean_acc_stats1)
        #w_data_noise_means = resps.t()[:, :, None] * data_noise_means
        #class_mean_acc_stats2 = \
        #    (class_s_mean @ w_data_noise_means.sum(dim=1).t()).t()
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
    inti_class_means = []
    for i in range(n_classes):
        r_mean = noise_std * torch.randn(len(mean), dtype=dtype, device=device)
        inti_class_means.append(r_mean)

    cov = torch.eye(len(mean), dtype=dtype, device=device) / prior_strength
    class_mean_priors, class_mean_posteriors = [], []
    for prior_mean, post_mean in zip(class_means, inti_class_means):
        class_mean_priors.append(NormalFullCovariancePrior(prior_mean, cov))
        class_mean_posteriors.append(NormalFullCovariancePrior(post_mean, cov))

    return MarginalPLDASet(
        normal,
        class_mean_priors,
        class_mean_posteriors
    )


__all__ = ['MarginalPLDASet']
