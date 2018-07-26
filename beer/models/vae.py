
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
from .. import nnet


##############################################
# Log-likelihood function for different VAE. #
##############################################

def _normal_log_likelihood(data, means, variances):
    distance_term = 0.5 * (data - means).pow(2) / variances
    precision_term = 0.5 * variances.log()
    llh =  (-distance_term - precision_term).sum(dim=-1)
    llh -= .5 * means.shape[-1] * math.log(2 * math.pi)
    return llh


def _bernoulli_log_likelihood(data, mean):
    epsilon = 1e-6
    per_pixel_bce = data * torch.log(epsilon + mean) + \
        (1.0 - data) * torch.log(epsilon + 1 - mean)
    return per_pixel_bce.sum(dim=-1)


def _beta_log_likelihood(data, alpha, beta):
    epsilon = 1e-6
    llh = (alpha - 1) * torch.log(epsilon + data) + \
        (beta - 1) * torch.log(epsilon + 1 - data) + \
        torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    return llh.sum(dim=-1)


llh_fns = {
    'normal': _normal_log_likelihood,
    'bernoulli': _bernoulli_log_likelihood,
    'beta': _beta_log_likelihood
}


################################
# Inverse AutoRegressive Flow #
###############################

class InverseAutoRegressiveFlow(torch.nn.Module):

    def __init__(self, nnet_flow):
        '''
        Args:
            nnet_flow (list): Sequence of transformation.
        '''
        super().__init__()
        self.nnet_flow = torch.nn.Sequential(*nnet_flow)

    def forward(self, mean, variance, flow_params, use_mean=False,
                stop_level=-1):
        if use_mean:
            noise = torch.zeros_like(mean)
        else:
            noise = torch.randn(*mean.shape, dtype=mean.dtype,
                                device=mean.device)

        # Initialize the flow
        feadim = mean.shape[1]
        flow = mean + variance.sqrt() * noise
        llh = -.5 * ((noise ** 2).sum(dim=-1) + feadim * math.log(2 * math.pi))
        llh = - .5 * torch.log(variance).sum(dim=-1)

        for i, flow_step in enumerate(self.nnet_flow):
            new_mean, new_variance = flow_step(flow, flow_params)
            flow = new_mean + torch.sqrt(new_variance) * flow
            llh += -.5 * torch.log(new_variance).sum(dim=-1)
            if stop_level >= 0 and i >= stop_level:
                break
        return llh, flow
        return avg_llh, samples


class VAE(BayesianModel):
    '''Variational Auto-Encoder (VAE).'''

    def __init__(self, encoder, decoder, latent_model, nflow, llh_fn):
        '''Initialize the VAE.

        Args:
            encoder (``MLPModel``): Encoder of the VAE.
            decoder (``MLPModel``): Decoder of the VAE.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
            llh_fn (function): Function to compute the log-likelihood.
        '''
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nflow = nflow
        self.llh_fn = llh_fn

    def _log_likelihood_from_states(self, data, states):
        params = list(self.decoder(states))
        for i, param in enumerate(params):
            params[i] = param
        return self.llh_fn(data, *params)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_model.mean_field_factorization()

    @staticmethod
    def sufficient_statistics(data):
        return data

    def forward(self, s_stats, kl_weight=1., use_mean=False, **kwargs):
        # For the case of the VAE, the sufficient statistics is just
        # the data itself. We just rename s_stats to avoid
        # confusion with the sufficient statistics of the latent model.
        data = s_stats

        # Sample states from the posterior distribution.
        mean, variance, flow_params = self.encoder(data)
        nflow_llh, nflow_samples = self.nflow(mean, variance, flow_params,
                                             use_mean)

        # Expected log-likelihood of the states given the prior model.
        latent_stats = self.latent_model.sufficient_statistics(nflow_samples)
        self.cache['latent_stats'] = latent_stats
        prior_llh = self.latent_model(latent_stats, **kwargs)

        # KL divergence posterior / prior.
        local_kl_div = nflow_llh - prior_llh

        # Return the log-likelihood of the model.
        retval = self._log_likelihood_from_states(data, nflow_samples)
        retval -= kl_weight * local_kl_div

        return retval

    def accumulate(self, _, parent_msg=None):
        latent_stats = self.cache['latent_stats']
        return self.latent_model.accumulate(latent_stats, parent_msg)


class VAEGlobalMeanVariance(VAE):
    '''Variational Auto-Encoder (VAE) with a global mean and
    (isostropic) covariance matrix parameters.

    '''

    def __init__(self, normal, encoder, decoder, latent_model, nflow):
        super().__init__(encoder, decoder, latent_model, nflow, None)
        self.normal = normal

    def _log_likelihood_from_states(self, data, states):
        dec_means = self.decoder(states)[0]
        centered_data = (data - dec_means)
        s_stats = self.normal.sufficient_statistics(centered_data)
        llh = self.normal(s_stats)
        self.cache['centered_s_stats'] = s_stats
        return llh

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_model.mean_field_factorization() + \
            self.normal.mean_field_factorization()

    def accumulate(self, _, parent_msg=None):
        latent_stats = self.cache['latent_stats']
        centered_s_stats = self.cache['centered_s_stats']
        return {
            **self.latent_model.accumulate(latent_stats),
            **self.normal.accumulate(centered_s_stats)
        }


#################
# VAE creation. #
#################

def create_probabbilistic_nnet(conf, dtype, device):
    nnet_blocks = list(nnet.neuralnetwork.create(conf['nnet_structure'],
                                                 dtype, device))
    nnet_blocks.append(nnet.problayers.create(conf['prob_layer']))
    return torch.nn.Sequential(*nnet_blocks).type(dtype).to(device)


def create_nflow(conf, dtype, device):
    flow_type = conf['type']
    nnet_flow = []
    if flow_type == 'InverseAutoRegressive':
        depth = conf['depth']
        for i in range(depth):
            nnet_flow.append(
                nnet.arnet.create_arnetwork(conf['iaf_block']).type(dtype).to(device)
            )
        return InverseAutoRegressiveFlow(nnet_flow).type(dtype).to(device)
    else:
        raise ValueError('Unsupported flow type: {}'.format(flow_type))

def create_vae(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    llh_fn = llh_fns[model_conf['llh_type']]
    latent_dim = model_conf['encoder']['prob_layer']['dim_out']
    encoder = create_probabbilistic_nnet(model_conf['encoder'], dtype, device)
    decoder = create_probabbilistic_nnet(model_conf['decoder'], dtype, device)
    latent_model = create_model_handle(model_conf['latent_model'],
                                       torch.zeros(latent_dim, dtype=dtype,
                                                   device=device),
                                       torch.ones(latent_dim, dtype=dtype,
                                                   device=device), create_model_handle)
    iaf = create_nflow(model_conf['normalizing_flow'], dtype, device)
    return VAE(encoder, decoder, latent_model, iaf, llh_fn)


def create_non_linear_subspace_model(model_conf, mean, variance,
                                     create_model_handle):
    dtype, device = mean.dtype, mean.device
    normal = create_model_handle(model_conf['normal_model'],
                                 mean, variance, create_model_handle)
    latent_dim = model_conf['encoder']['prob_layer']['dim_out']
    encoder = create_probabbilistic_nnet(model_conf['encoder'], dtype, device)
    decoder = create_probabbilistic_nnet(model_conf['decoder'], dtype, device)
    latent_model = create_model_handle(model_conf['latent_model'],
                                       torch.zeros(latent_dim, dtype=dtype,
                                                   device=device),
                                       torch.ones(latent_dim, dtype=dtype,
                                                   device=device), create_model_handle)
    iaf = create_nflow(model_conf['normalizing_flow'], dtype, device)
    return VAEGlobalMeanVariance(normal, encoder, decoder, latent_model, iaf)


__all__ = [
    'VAE',
    'VAEGlobalMeanVariance',
]
