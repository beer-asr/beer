
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import torch
from .bayesmodel import BayesianModel


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

        # Temporary cache variable(s) used during training.
        self._s_stats = None

    @staticmethod
    def sufficient_statistics(data):
        return data

    def forward(self, s_stats, latent_variables=None):
        # For the case of the VAE, the sufficient statistics is just
        # the data itself. We just rename the s_stats to avoid
        # confusion with the sufficient statistics of the latent model.
        data = s_stats

        enc_state = self.encoder(data)
        mean, var = enc_state.mean, enc_state.var
        self._s_stats = \
            self.latent_model.sufficient_statistics_from_mean_var(mean, var)
        exp_np_params = self.latent_model.expected_natural_params(
            mean.detach(), var.detach(), latent_variables=latent_variables,
            nsamples=self.nsamples)
        samples = mean + torch.sqrt(var) * torch.randn(self.nsamples,
                                                       data.size(0),
                                                       mean.size(1))
        self.cache['kl_divergence'] = enc_state.kl_div(exp_np_params)
        llh = self.decoder(samples).log_likelihood(data)
        return llh

    def local_kl_div_posterior_prior(self):
        return self.cache['kl_divergence'] + \
            self.latent_model.local_kl_div_posterior_prior()

    #def evaluate(self, data):
    #    'Convenience function mostly for plotting and debugging.'
    #    torch_data = Variable(torch.from_numpy(data).float())
    #    state = self(torch_data, sampling=sampling)
    #    loss, llh, kld = self.loss(torch_data, state)
    #    return -loss, llh, kld, state['encoder_state'].mean, \
    #        state['encoder_state'].std_dev ** 2

    def accumulate(self, _, parent_msg=None):
        return self.latent_model.accumulate(self._s_stats, parent_msg)

__all__ = ['VAE']
