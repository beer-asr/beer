
'''Bayesian Subspace models.'''

import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel


########################################################################
# Probabilistic Principal Component Analysis (PPCA)
########################################################################

class PPCA(BayesianModel):
    'Probabilistic Principal Component Analysis (PPCA).'

    def __init__(self, prior_prec, posterior_prec, prior_mean, posterior_mean,
                 prior_subspace, posterior_subspace, subspace_dim):
        '''Initialize the PPCA model.

        Args:
            prior_prec (``GammaPrior``): Prior over the global precision.
            posterior_prec (``GammaPrior``): Posterior over the global
                precision.
            prior_mean (``NormalIsotropicCovariancePrior``): Prior over
                the global mean.
            posterior_mean (``NormalIsotropicCovariancePrior``):
                Posterior over the global mean.
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


    @property
    def mean(self):
        _, mean = self.mean_param.expected_value(concatenated=False)
        return mean

    @property
    def precision(self):
        return self.precision_param.expected_value()[1]

    @property
    def subspace(self):
        post = self.subspace_param.posterior
        _, exp_value2 =  self.subspace_param.expected_value(concatenated=False)
        return exp_value2

    def latent_posterior(self, stats):
        data = stats[:, 1:-1]
        mean = self.mean
        subspace_cov, subspace_mean = \
            self.subspace_param.expected_value(concatenated=False)
        prec = self.precision
        lposterior_cov = torch.inverse(
            torch.eye(self._subspace_dim).type(stats.type()) + prec * subspace_cov)
        lposterior_means = prec * lposterior_cov @ subspace_mean @ (data - mean).t()
        return lposterior_means, lposterior_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data,
                          torch.ones(len(data), 1).type(data.type())], dim=-1)

    def forward(self, s_stats, latent_variables=None):
        feadim = .25 * s_stats.size(1)
        exp_llh = s_stats @ self.mean_prec_param.expected_value()
        exp_llh -= .5 * feadim * math.log(2 * math.pi)
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        return {self.mean_prec_param: s_stats.sum(dim=0)}

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([torch.sum(mean ** 2 + var, dim=1).view(-1, 1), mean,
                          torch.ones(len(mean), 1).type(mean.type())], dim=-1)

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
        nparams = self.mean_prec_param.expected_value
        ones = torch.ones(s_stats.size(0), nparams.size(0)).type(s_stats.type())
        return ones * nparams
