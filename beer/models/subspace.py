
'''Bayesian Subspace models.'''

import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel
from ..expfamilyprior import GammaPrior
from ..expfamilyprior import NormalIsotropicCovariancePrior
from ..expfamilyprior import MatrixNormalPrior


########################################################################
# Probabilistic Principal Component Analysis (PPCA)
########################################################################

class PPCA(BayesianModel):
    'Probabilistic Principal Component Analysis (PPCA).'

    def __init__(self, prior_mean, posterior_mean, prior_prec, posterior_prec,
                 prior_subspace, posterior_subspace, subspace_dim):
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
        self._subspace_dim = subspace_dim

    @classmethod
    def create(cls, mean, precision, subspace, pseudo_counts=1.):
        '''Create a Probabilistic Principal Ccomponent model.

        Args:
            mean (``torch.Tensor``): Mean of the model.
            precision (float): Global precision of the model.
            subspace (``torch.Tensor[subspace_dim, dim]``): Mean of the
                Subspace matrix. `subspace_dim` and `dim` are the
                dimension of the subspace and the data respectively.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.

        Returns:
            :any:`PPCA`
        '''
        shape = torch.tensor([pseudo_counts]).type(mean.type())
        rate = torch.tensor([pseudo_counts / float(precision)]).type(mean.type())
        prior_prec = GammaPrior(shape, rate)
        posterior_prec = GammaPrior(shape, rate)
        variance = torch.tensor([1. / float(pseudo_counts)]).type(mean.type())
        prior_mean = NormalIsotropicCovariancePrior(mean, variance)
        posterior_mean = NormalIsotropicCovariancePrior(mean, variance)

        cov = torch.eye(subspace.size(0)).type(mean.type()) / pseudo_counts
        prior_subspace = MatrixNormalPrior(subspace, cov)
        posterior_subspace = MatrixNormalPrior(subspace, cov)

        return cls(prior_mean, posterior_mean, prior_prec, posterior_prec,
                   prior_subspace, posterior_subspace, subspace.size(0))

    @property
    def mean(self):
        _, mean = self.mean_param.expected_value(concatenated=False)
        return mean

    @property
    def precision(self):
        return self.precision_param.expected_value()[1]

    @property
    def subspace(self):
        _, exp_value2 =  self.subspace_param.expected_value(concatenated=False)
        return exp_value2

    def latent_posterior(self, stats):
        data = stats[:, 1:]
        mean = self.mean
        subspace_cov, subspace_mean = \
            self.subspace_param.expected_value(concatenated=False)
        prec = self.precision
        lposterior_cov = torch.inverse(
            torch.eye(self._subspace_dim).type(stats.type()) + prec * subspace_cov)
        lposterior_means = prec * lposterior_cov @ subspace_mean @ (data - mean).t()
        return lposterior_means.t(), lposterior_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data],
                          dim=-1)

    def forward(self, s_stats, latent_variables=None):
        feadim = s_stats.size(1) - 1

        if latent_variables is not None:
            l_means = latent_variables
            l_quad = l_means[:, :, None] * l_means[:, None, :]
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        s_quad, s_mean = self.subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)

        exp_llh = torch.zeros(len(s_stats)).type(s_stats.type())
        exp_llh += -.5 * feadim * math.log(2 * math.pi)
        exp_llh += .5 * feadim * log_prec
        exp_llh += -.5 * prec * s_stats[:, 0]
        exp_llh += torch.sum(prec * \
            (s_mean.t() @ l_means.t() + m_mean[:, None]) * s_stats[:, 1:].t(), dim=0)
        exp_llh += -.5 * prec * torch.sum(l_quad * s_quad.view(-1), dim=1)
        exp_llh += - prec * l_means @ s_mean @ m_mean
        exp_llh += -.5 * prec * m_quad
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([torch.sum(mean ** 2 + var, dim=1).view(-1, 1), mean],
                         dim=1)

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
                natural parameters.

        Returns:
            (Tensor): Expected value of the natural parameters.

        '''
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)
        l_means, l_cov = self.latent_posterior(s_stats)
        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        s_quad, s_mean = self.subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)
        l_quad = l_cov + \
            torch.sum(l_means[:, :, None] * l_means[:, None, :], dim=0)

        np1 = -.5 * prec * torch.ones(len(s_stats),
                                      mean.size(1)).type(mean.type())
        np2 = prec * (l_means @ s_mean + m_mean)
        np3 = torch.zeros(len(s_stats), mean.size(1)).type(mean.type())
        np3 += -.5 * prec * torch.trace(s_quad @ l_quad)
        np3 += - (prec * l_means @ s_mean @ m_mean).reshape(-1, 1)
        np3 += -.5 * prec * m_quad
        np4 = -.5 * log_prec * torch.ones(s_stats.size(0),
                                          mean.size(1)).type(mean.type())
        return torch.cat([np1, np2, np3, np4], dim=1)
