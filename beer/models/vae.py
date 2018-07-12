
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import copy
import math
import torch
from .bayesmodel import BayesianModel
from .normal import NormalDiagonalCovariance
from .normal import NormalIsotropicCovariance
from ..utils import sample_from_normals
from . import nnet


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


def _normal_log_likelihood(data, means, variances):
    distance_term = 0.5 * (data - means).pow(2) / variances
    precision_term = 0.5 * variances.log()
    llh =  (-distance_term - precision_term).sum(dim=-1).mean(dim=0)
    llh -= .5 * means.shape[-1] * math.log(2 * math.pi)
    return llh


def _bernoulli_log_likelihood(data, mean):
    epsilon = 1e-6
    per_pixel_bce = data * torch.log(epsilon + mean) + \
        (1.0 - data) * torch.log(epsilon + 1 - mean)
    return per_pixel_bce.sum(dim=-1).mean(dim=0)


def _beta_log_likelihood(data, alpha, beta):
    epsilon = 1e-6
    llh = (alpha - 1) * torch.log(epsilon + data) + \
        (beta - 1) * torch.log(epsilon + 1 - data) + \
        torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    return llh.sum(dim=-1).mean(dim=0)



class VAE(BayesianModel):
    '''Variational Auto-Encoder (VAE).'''

    def __init__(self, encoder, decoder, latent_model):
        '''Initialize the VAE.

        Args:
            encoder (``MLPModel``): Encoder of the VAE.
            decoder (``MLPModel``): Decoder of the VAE.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
        '''
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model

    def _estimate_prior(self, means, variances, nsamples, **kwargs):
        exp_np_params, s_stats = self.latent_model.expected_natural_params(
            means.detach(), variances.detach(), nsamples=nsamples,
            **kwargs)
        self.cache['latent_stats'] = s_stats
        return exp_np_params

    def _compute_local_kl_div(self, means, variances, exp_np_params):
        exp_s_stats = \
            NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
                means, variances)
        nparams = _normal_diag_natural_params(means, variances)
        self.cache['kl_divergence'] = \
            ((nparams - exp_np_params) * exp_s_stats).sum(dim=-1)

    def _expected_llh(self, data, means, variances, nsamples):
        len_data = len(data)
        samples = sample_from_normals(means, variances, nsamples)
        samples = samples.view(nsamples * len_data, -1)
        dec_means, dec_variances = self.decoder(samples)
        dec_means = dec_means.view(nsamples, len_data, -1)
        dec_variances = dec_variances.view(nsamples, len_data, -1)
        return _normal_log_likelihood(data, dec_means, dec_variances)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @property
    def grouped_parameters(self):
        return self.latent_model.grouped_parameters

    @staticmethod
    def sufficient_statistics(data):
        return data

    def float(self):
        return self.__class__(
            self.encoder.float(),
            self.decoder.float(),
            self.latent_model.float(),
        )

    def double(self):
        return self.__class__(
            self.encoder.double(),
            self.decoder.double(),
            self.latent_model.double(),
        )

    def to(self, device):
        return self.__class__(
            self.encoder.to(device),
            self.decoder.to(device),
            self.latent_model.to(device),
        )

    def non_bayesian_parameters(self):
        retval = [param.data for param in self.encoder.parameters()]
        retval += [param.data for param in self.decoder.parameters()]
        return retval

    def set_non_bayesian_parameters(self, new_params):
        self.encoder = copy.deepcopy(self.encoder)
        self.decoder = copy.deepcopy(self.decoder)
        n_params_enc = len(list(self.encoder.parameters()))
        params_enc, params_dec = new_params[:n_params_enc], new_params[n_params_enc:]
        for param, new_p_data in zip(self.encoder.parameters(), params_enc):
            param.data = new_p_data
        for param, new_p_data in zip(self.decoder.parameters(), params_dec):
            param.data = new_p_data

    def forward(self, s_stats, nsamples=1, **kwargs):
        # For the case of the VAE, the sufficient statistics is just
        # the data itself. We just rename s_stats to avoid
        # confusion with the sufficient statistics of the latent model.
        data = s_stats
        means, variances = self.encoder(data)
        exp_np_params = self._estimate_prior(means, variances, nsamples, **kwargs)
        self._compute_local_kl_div(means, variances, exp_np_params)
        return self._expected_llh(data, means, variances, nsamples)

    def local_kl_div_posterior_prior(self, parent_msg=None):
        return self.cache['kl_divergence'] + \
            self.latent_model.local_kl_div_posterior_prior()

    def accumulate(self, _, parent_msg=None):
        latent_stats = self.cache['latent_stats']
        self.clear_cache()
        return self.latent_model.accumulate(latent_stats, parent_msg)


class VAEGlobalMeanCovariance(VAE):
    '''Variational Auto-Encoder (VAE) with a global mean and
    (isostropic) covariance matrix parameters.

    '''

    def __init__(self, normal, encoder, decoder, latent_model):
        super().__init__(encoder, decoder, latent_model)
        self.normal = normal

    def _expected_llh(self, data, means, variances, nsamples):
        samples = sample_from_normals(means, variances, nsamples)
        samples = samples.view(nsamples * len(data), -1)
        dec_means = self.decoder(samples).view(nsamples, len(data), -1)
        centered_data = (data[None] - dec_means).view(nsamples * len(data), -1)
        s_stats = self.normal.sufficient_statistics(centered_data)
        llh = self.normal(s_stats).view(nsamples, len(data), -1).mean(dim=0)
        s_stats = s_stats.view(nsamples, len(data), -1).mean(dim=0)
        self.cache['centered_s_stats'] = s_stats
        return llh

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    # Most of the BayesianModel interface is implemented in the parent
    # class VAE.

    @property
    def grouped_parameters(self):
        groups = self.normal.grouped_parameters
        groups += self.latent_model.grouped_parameters
        return groups

    def float(self):
        return self.__class__(
            self.normal.float(),
            self.encoder.float(),
            self.decoder.float(),
            self.latent_model.float(),
        )

    def double(self):
        return self.__class__(
            self.normal.double(),
            self.encoder.double(),
            self.decoder.double(),
            self.latent_model.double(),
        )

    def to(self, device):
        return self.__class__(
            self.normal.to(device),
            self.encoder.to(device),
            self.decoder.to(device),
            self.latent_model.to(device),
        )

    def accumulate(self, _, parent_msg=None):
        latent_stats = self.cache['latent_stats']
        centered_s_stats = self.cache['centered_s_stats']
        self.clear_cache()
        return {
            **self.latent_model.accumulate(latent_stats),
            **self.normal.accumulate(centered_s_stats)
        }


class BernoulliVAE(VAE):
    '''Variational Auto-Encoder (VAE) for discrete data.

    '''

    def _expected_llh(self, data, means, variances, nsamples):
        samples = sample_from_normals(means, variances, nsamples)
        samples = samples.view(nsamples * len(data), -1)
        dec_means = self.decoder(samples).view(nsamples, len(data), -1)
        return _bernoulli_log_likelihood(data, dec_means)


class BetaVAE(VAE):
    '''Variational Auto-Encoder (VAE) for continuous data ranging from
    0 to 1.

    '''

    def _expected_llh(self, data, means, variances, nsamples):
        samples = sample_from_normals(means, variances, nsamples)
        samples = samples.view(nsamples * len(data), -1)
        dec_alphas, dec_betas = self.decoder(samples)
        dec_alphas = dec_alphas.view(nsamples, len(data), -1)
        dec_betas = dec_betas.view(nsamples, len(data), -1)
        return _beta_log_likelihood(data, dec_alphas, dec_betas)


def create_normal_vae(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    variables = {'<feadim>': len(mean)}
    latent_dim = model_conf['encoder']['dim_output_normal_layer']
    encoder = nnet.create_encoder(model_conf['encoder'], dtype, device,
                                  variables)
    decoder = nnet.create_normal_decoder(model_conf['decoder'], dtype, device,
                                         variables)
    normal = create_model_handle(model_conf['normal_model'],
                                 mean, variance, create_model_handle)
    latent_model = create_model_handle(model_conf['latent_model'],
                                       torch.zeros(latent_dim, dtype=dtype,
                                                   device=device),
                                       torch.ones(latent_dim, dtype=dtype,
                                                   device=device), create_model_handle)
    return VAEGlobalMeanCovariance(normal, encoder, decoder, latent_model)


def create_bernoulli_vae(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    variables = {'<feadim>': len(mean)}
    latent_dim = model_conf['encoder']['dim_output_normal_layer']
    encoder = nnet.create_encoder(model_conf['encoder'], dtype, device,
                                  variables)
    decoder = nnet.create_bernoulli_decoder(model_conf['decoder'], dtype,
                                            device, variables)
    latent_model = create_model_handle(model_conf['latent_model'],
                                       torch.zeros(latent_dim, dtype=dtype,
                                                   device=device),
                                       torch.ones(latent_dim, dtype=dtype,
                                                   device=device), create_model_handle)
    return BernoulliVAE(encoder, decoder, latent_model)


def create_beta_vae(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    variables = {'<feadim>': len(mean)}
    latent_dim = model_conf['encoder']['dim_output_normal_layer']
    encoder = nnet.create_encoder(model_conf['encoder'], dtype, device,
                                  variables)
    decoder = nnet.create_beta_decoder(model_conf['decoder'], dtype,
                                       device, variables)
    latent_model = create_model_handle(model_conf['latent_model'],
                                       torch.zeros(latent_dim, dtype=dtype,
                                                   device=device),
                                       torch.ones(latent_dim, dtype=dtype,
                                                   device=device), create_model_handle)
    return BetaVAE(encoder, decoder, latent_model)


__all__ = [
    'VAE',
    'VAEGlobalMeanCovariance',
    'BernoulliVAE'
]
