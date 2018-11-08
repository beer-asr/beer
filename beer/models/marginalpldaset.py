'''Bayesian Subspace models.'''


import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .modelset import BayesianModelSet
from .normal import Normal
from .normalset import NormalSetElement
from ..priors import NormalFullCovariancePrior
from ..priors import WishartPrior
from ..utils import make_symposdef


class MarginalPLDASet(BayesianModelSet):
    'PLDA set where the noise/class subspace are marginalized.'

    @staticmethod
    def create(global_mean, noise_cov, class_cov, size, noise_std=0.,
               prior_strength=1.):
        dim, dtype, device = len(global_mean), global_mean.dtype, \
            global_mean.device

        # Ensure the the class covariance is full.
        if len(class_cov.shape) == 1:
            if class_cov.shape[0] == 1:
                f_class_cov = class_cov * torch.eye(dim, dtype=dtype,
                                                    device=device)
            else:
                f_class_cov = class_cov.diag()
        else:
            f_class_cov = class_cov
        p_strength = torch.tensor(prior_strength, dtype=dtype, device=device)

        # The base normal model contain the prior over the global mean
        # and the precision matrix (shared accross clusters).
        normal =  Normal.create(global_mean, noise_cov,
                                p_strength,
                                cov_type='full')

        # Prior over the class precision matrix.
        dof = torch.tensor(prior_strength + dim - 1, dtype=dtype, device=device)
        scale_matrix = f_class_cov.inverse() / dof
        prior_class_prec = WishartPrior(scale_matrix, dof)
        posterior_class_prec = WishartPrior(scale_matrix, dof)

        # Class means
        init_class_means = []
        for i in range(size):
            r_mean = noise_std * torch.randn(dim, dtype=dtype, device=device)
            init_class_means.append(r_mean)

        class_mean_priors, class_mean_posteriors = [], []
        class_mean_prior = NormalFullCovariancePrior(
                torch.zeros_like(global_mean), p_strength,
                posterior_class_prec)
        for post_mean in init_class_means:
            class_mean_priors.append(class_mean_prior)
            class_mean_posteriors.append(NormalFullCovariancePrior(post_mean,
                                         p_strength,
                                         posterior_class_prec))

        return MarginalPLDASet(
            normal,
            class_mean_priors,
            class_mean_posteriors,
            prior_class_prec,
            posterior_class_prec
        )


    def __init__(self, normal, prior_means, posterior_means, prior_class_prec,
                 posterior_class_prec):
        super().__init__()
        self.normal = normal
        self.class_mean_params = BayesianParameterSet([
            BayesianParameter(prior, posterior)
            for prior, posterior in zip(prior_means, posterior_means)
        ])
        self.class_prec_param = BayesianParameter(prior_class_prec,
                                                  posterior_class_prec)
        #self.class_prec_param.register_callback(self.on_class_cov_update)

    def on_class_cov_update(self):
        raise NotImplementedError

    @property
    def mean(self):
        return self.normal.mean

    @property
    def cov(self):
        return self.normal.cov

    @property
    def class_means(self):
        means = [mean_param.expected_value()
                 for mean_param in self.class_mean_params]
        return torch.stack(means)

    @property
    def class_covs(self):
        scales = [mean_param.posterior.to_std_parameters()[1]
                  for mean_param in self.class_mean_params]
        cov = self.class_prec_param.expected_value().inverse()
        covs = [cov * scale for scale in scales]
        return torch.stack(covs)

    @property
    def class_cov(self):
        np1, np2 = self.class_cov_param.expected_value(concatenated=False)
        return torch.inverse(np1)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return [
            *self.normal.mean_field_factorization(),
            [*self.class_mean_params],
            #[self.class_prec_param],
        ]

    def sufficient_statistics(self, data):
        # We need the raw data to accumualte the s. statistics.
        self.cache['data'] = data

        means = self.class_means
        covs = self.class_covs
        data_means = data[:, None, :] - means[None]
        data_mean_quad = data_means[:, :, :, None] * data_means[:, :, None, :]
        return torch.cat([
            -.5 * (data_mean_quad + covs[None]).view(len(data), len(self), -1),
            data_means,
            -.5 * torch.ones(len(data), len(self), 1, dtype=data.dtype,
                       device=data.device),
            .5 * torch.ones(len(data), len(self), 1, dtype=data.dtype,
                       device=data.device),
        ], dim=-1)

    def expected_log_likelihood(self, s_stats):
        feadim = len(self.normal.mean)
        nparams = self.normal.mean_precision.expected_natural_parameters()
        exp_llh = s_stats @ nparams
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, resps):
        dtype = resps.dtype
        device = resps.device
        data = self.cache['data']
        mean, prec = self.normal.mean_precision.expected_value()
        prec_data_mean = (data - mean)

        acc_stats = self.normal.accumulate(
            (s_stats * resps[:, :, None]).sum(dim=1))

        # Accumulate the statistics for the class means.
        import pdb; pdb.set_trace()
        acc_resps = resps.sum(dim=0)
        acc_prec_data_mean = (prec_data_mean[:, :, None] @ resps[:, None, :]).sum(dim=0).t()
        for i, mean_param in enumerate(self.class_mean_params):
            class_mean_acc_stats = {
                mean_param: torch.cat([
                    0 * acc_prec_data_mean[i],
                    0 * (-.5 * acc_resps[i]).view(1),
                ])
            }
            acc_stats.update(class_mean_acc_stats)

        # Accumulate statistics for the class cov (i.e. the between
        # class covariance matrix).
        prior_mean = self.class_mean_params[0].prior.expected_value()

        class_means_quad, class_mean_mean = [], []
        for mean, cov in zip(self.class_means, self.class_covs):
            quad_mean = cov + torch.ger(mean, mean)
            class_means_quad.append(quad_mean)
            mean_prior_mean = torch.ger(mean, prior_mean)
            class_mean_mean.append(mean_prior_mean + mean_prior_mean.t() )
        class_means_quad = torch.stack(class_means_quad)
        class_means_mean = torch.stack(class_mean_mean)
        prior_mean_mean = torch.ger(prior_mean, prior_mean)
        stats = class_means_quad - class_means_mean + prior_mean_mean
        stats = make_symposdef(stats.sum(dim=0))
        class_cov_acc_stats = {
            self.class_prec_param: torch.cat([
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


__all__ = ['MarginalPLDASet']
