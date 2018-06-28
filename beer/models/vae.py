
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import math
import torch
from .bayesmodel import BayesianModel
from .normal import NormalDiagonalCovariance
from ..utils import sample_from_normals


def _normal_diag_natural_params(mean, var):
    '''Transform the standard parameters of a Normal (diag. cov.) into
    their canonical forms.

    Note:
        The (negative) log normalizer is appended to it.

    '''
    return torch.cat([
        -1. / (2 * var),
        mean / var,
        -(mean ** 2) / (2 * var),
        -.5 * torch.log(var)
    ], dim=-1)


def _log_likelihood(data, means, variances):
        distance_term = 0.5 * (data - means).pow(2) / variances
        precision_term = 0.5 * variances.log()
        llh =  (-distance_term - precision_term).sum(dim=-1)
        llh -= .5 * means.shape[-1] * math.log(2 * math.pi)
        return llh


class VAE(BayesianModel):
    '''Variational Auto-Encoder (VAE).'''

    def __init__(self, encoder, decoder, latent_model, nsamples):
        '''Initialize the VAE.

        Args:
            encoder (``MLPModel``): Encoder of the VAE.
            decoder (``MLPModel``): Decoder of the VAE.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
            nsamples (int): Number of samples to approximate the
                expectation of the log-likelihood.

        '''
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nsamples = nsamples

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return data

    def float(self):
        return self.__class__(
            self.encoder.float(),
            self.decoder.float(),
            self.latent_model.float(),
            self.nsamples
        )

    def double(self):
        return self.__class__(
            self.encoder.double(),
            self.decoder.double(),
            self.latent_model.double(),
            self.nsamples
        )

    def to(self, device):
        return self.__class__(
            self.encoder.to(device),
            self.decoder.to(device),
            self.latent_model.to(device),
            self.nsamples
        )

    def forward(self, s_stats, **kwargs):
        # For the case of the VAE, the sufficient statistics is just
        # the data itself. We just rename s_stats to avoid
        # confusion with the sufficient statistics of the latent model.
        data = s_stats

        # Compute the posterior of the latent variables.
        means, variances = self.encoder(data)

        # Compute the prior over the latent variables.
        exp_np_params, s_stats = self.latent_model.expected_natural_params(
            means.detach(), variances.detach(), nsamples=self.nsamples,
            **kwargs)
        self.cache['latent_stats'] = s_stats

        # (local) KL divergence posterior / prior.
        exp_s_stats = \
            NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
                means, variances)
        nparams = _normal_diag_natural_params(means, variances)
        self.cache['kl_divergence'] = \
            ((nparams - exp_np_params) * exp_s_stats).sum(dim=-1)

        # Expected value of the log-likelihood per frame using the
        # re-parameterization trick.
        samples = sample_from_normals(means, variances, self.nsamples)
        dec_means, dec_variances = \
            self.decoder(samples.view(self.nsamples * len(data), -1))
        dec_means, dec_variances = \
            dec_means.view(self.nsamples, len(data), -1), \
            dec_variances.view(self.nsamples, len(data), -1)
        return _log_likelihood(data, dec_means, dec_variances).mean(dim=0)

    def local_kl_div_posterior_prior(self, parent_msg=None):
        return self.cache['kl_divergence'] + \
            self.latent_model.local_kl_div_posterior_prior()

    def accumulate(self, _, parent_msg=None):
        latent_stats = self.cache['latent_stats']
        self.clear_cache()
        return self.latent_model.accumulate(latent_stats, parent_msg)


class VAEGlobalMeanVar(BayesianModel):
    '''Variational Auto-Encoder (VAE) with a global mean and variance
    parameters. Strictly speaking this model is the non-linear version
    of the Probabilistics Principle Component Analysis.

    '''

    def __init__(self, normal, encoder, decoder, latent_model, nsamples):
        '''Initialize the VAE.

        Args:
            normal (:any:`beer.NormalDiagonalCovariance`): Main
                component of the model.
            encoder (``MLPModel``): Encoder of the VAE.
            decoder (``MLPModel``): Decoder of the VAE.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
            nsamples (int): Number of samples to approximate the
                expectation of the log-likelihood.

        '''
        super().__init__()
        self.normal = normal
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nsamples = nsamples

    @classmethod
    def create(cls, mean, diag_cov, encoder, decoder, latent_model, nsamples,
               pseudo_counts=1.):
        '''Create a :any:`NormalDiagonalCovariance`.

        Args:
            mean (``torch.Tensor``): Mean of the Normal to create.
            diag_cov (``torch.Tensor``): Diagonal of the covariance
                matrix of the Normal to create.
            encoder (``MLPModel``): Encoder of the VAE.
            decoder (``MLPModel``): Decoder of the VAE.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
            nsamples (int): Number of samples to approximate the
                expectation of the log-likelihood.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.

        Returns:
            :any:`VAEGlobalMeanVar`

        '''
        normal = NormalDiagonalCovariance.create(mean, diag_cov, pseudo_counts)
        return cls(normal, encoder, decoder, latent_model, nsamples)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return data

    def float(self):
        return self.__class__(
            self.normal.float(),
            self.encoder.float(),
            self.decoder.float(),
            self.latent_model.float(),
            self.nsamples
        )

    def double(self):
        return self.__class__(
            self.normal.double(),
            self.encoder.double(),
            self.decoder.double(),
            self.latent_model.double(),
            self.nsamples
        )

    def to(self, device):
        return self.__class__(
            self.normal.to(device),
            self.encoder.to(device),
            self.decoder.to(device),
            self.latent_model.to(device),
            self.nsamples
        )

    def forward(self, s_stats, **kwargs):
        # For the case of the VAE, the sufficient statistics is just
        # the data itself. We just rename s_stats to avoid
        # confusion with the sufficient statistics of the latent model.
        data = s_stats

        # Compute the posterior of the latent variables.
        means, variances = self.encoder(data)

        # Compute the prior over the latent variables.
        exp_np_params, s_stats = self.latent_model.expected_natural_params(
            means.detach(), variances.detach(), nsamples=self.nsamples,
            **kwargs)
        self.cache['latent_stats'] = s_stats

        # (local) KL divergence posterior / prior.
        exp_s_stats = \
            NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
                means, variances)
        nparams = _normal_diag_natural_params(means, variances)
        self.cache['kl_divergence'] = \
            ((nparams - exp_np_params) * exp_s_stats).sum(dim=-1)

        # Expected value of the log-likelihood per frame using the
        # re-parameterization trick.
        samples = sample_from_normals(means, variances, self.nsamples)
        dec_means = self.decoder(samples.view(self.nsamples * len(data), -1))
        dec_means = dec_means.view(self.nsamples, len(data), -1)
        centered_data = data[None] - dec_means
        centered_s_stats = \
            self.normal.sufficient_statistics(centered_data).mean(dim=0)
        self.cache['centered_s_stats'] = centered_s_stats
        return self.normal(centered_s_stats)

    def local_kl_div_posterior_prior(self, parent_msg=None):
        return self.cache['kl_divergence'] + \
            self.latent_model.local_kl_div_posterior_prior()

    def accumulate(self, _, parent_msg=None):
        latent_stats = self.cache['latent_stats']
        centered_s_stats = self.cache['centered_s_stats']
        self.clear_cache()
        return {
            **self.latent_model.accumulate(latent_stats),
            **self.normal.accumulate(centered_s_stats)
        }



__all__ = ['VAE', 'VAEGlobalMeanVar']
