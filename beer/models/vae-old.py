
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import torch
from .bayesmodel import BayesianModel


class VAE(BayesianModel):
    '''Variational Auto-Encoder (VAE).'''

    def __init__(self, encoder, encoder_problayer, decoder,  decoder_problayer,
                 latent_model):
        '''Initialize the VAE.

        Args:
            encoder (``torch.nn.Module``): Encoder of the VAE.
            encoder_problayer (:any:`beer.nnet.ProbabilisticLayer`):
                Layer to transform the output of the encoder into a
                probability distribution.
            decoder (``torch.nn.Module``): Decoder of the VAE.
            decoder_problayer (:any:`beer.nnet.ProbabilisticLayer`):
                Layer to transform the output of the decoder into a
                probability distribution.
            latent_model(``BayesianModel``): Bayesian Model
                for the prior over the latent space.
        '''
        super().__init__()
        self.encoder = encoder
        self.encoder_problayer = encoder_problayer
        self.decoder = decoder
        self.decoder_problayer = decoder_problayer
        self.latent_model = latent_model

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_model.mean_field_factorization()

    @staticmethod
    def sufficient_statistics(data):
        #For the VAE, this is just the idenity function
        return data

    def expected_log_likelihood(self, stats, kl_weight=1., use_mean=False,
                                **kwargs):
        encoder_states = self.encoder(stats)
        posterior_params = self.encoder_problayer(encoder_states)
        samples, post_llh = self.encoder_problayer.samples_and_llh(
                posterior_params, use_mean)

        # Per-frame KL divergence between the (approximate) posterior
        # and the prior.
        latent_stats = self.latent_model.sufficient_statistics(samples)
        prior_llh = self.latent_model.expected_log_likelihood(latent_stats,
                                                              **kwargs)
        kl_divs = post_llh - prior_llh

        # Log-likelihood.
        decoder_states = self.decoder(samples)
        llh_params = self.decoder_problayer(decoder_states)
        llhs = self.decoder_problayer.log_likelihood(stats, llh_params)

        # Store the statistics of the latent model to compute its
        # gradients
        self.cache['latent_stats'] = latent_stats

        return llhs - kl_weight * kl_divs

    def marginal_log_likelihood(self, data, kl_weight=1., use_mean=False,
                                **kwargs):
        encoder_states = self.encoder(data)
        posterior_params = self.encoder_problayer(encoder_states)
        samples, post_llh = self.encoder_problayer.samples_and_llh(
            posterior_params, use_mean)

        # Per-frame KL divergence between the (approximate) posterior
        # and the prior.
        latent_stats = self.latent_model.sufficient_statistics(samples)
        prior_llh = self.latent_model.marginal_log_likelihood(latent_stats,
                                                              **kwargs)
        kl_divs = post_llh - prior_llh

        # Log-likelihood.
        decoder_states = self.decoder(samples)
        llh_params = self.decoder_problayer(decoder_states)
        llhs = self.decoder_problayer.log_likelihood(data, llh_params)

        # Store the statistics of the latent/likelihood model to
        # compute their gradients.
        self.cache['latent_stats'] = latent_stats.detach()
        return llhs - kl_weight * kl_divs

    def accumulate(self, _):
        latent_stats = self.cache['latent_stats']
        return self.latent_model.accumulate(latent_stats)


class VAEGlobalMeanVariance(VAE):
    '''Variational Auto-Encoder (VAE) with a global mean and
    (isostropic) covariance matrix parameters.

    '''

    def __init__(self, encoder, encoder_problayer, decoder,
                 normal, latent_model):
        super().__init__(encoder, encoder_problayer, decoder, None,
                         latent_model)
        self.normal = normal

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_model.mean_field_factorization() + \
            self.normal.mean_field_factorization()

    def expected_log_likelihood(self, data, kl_weight=1., use_mean=False,
                                nsamples=1, **kwargs):
        encoder_states = self.encoder(data)
        posterior_params = self.encoder_problayer(encoder_states)
        s_llhs = []
        for l in range(nsamples):
            samples, post_llh = self.encoder_problayer.samples_and_llh(
                posterior_params, use_mean)

            # Per-frame KL divergence between the (approximate) posterior
            # and the prior.
            latent_stats = self.latent_model.sufficient_statistics(samples)
            prior_llh = self.latent_model.expected_log_likelihood(latent_stats,
                                                                   **kwargs)
            kl_divs = post_llh - prior_llh

            decoder_means = self.decoder(samples)
            centered_data = (data - decoder_means)
            centered_stats = self.normal.sufficient_statistics(centered_data)
            llhs = self.normal.expected_log_likelihood(centered_stats)

            # Store the statistics of the latent/likelihood model to
            # compute their gradients.
            self.cache['latent_stats'] = latent_stats.detach()
            self.cache['centered_stats'] = centered_stats.detach()
            s_llhs.append((llhs - kl_weight * kl_divs).view(-1, 1))
        return torch.cat(s_llhs, dim=-1).mean(dim=-1)

    def marginal_log_likelihood(self, data, kl_weight=1., use_mean=False,
                                nsamples=1, **kwargs):
        encoder_states = self.encoder(data)
        posterior_params = self.encoder_problayer(encoder_states)
        
        s_llhs = []
        for l in range(nsamples):
            samples, post_llh = self.encoder_problayer.samples_and_llh(
                posterior_params, use_mean)

            # Per-frame KL divergence between the (approximate) posterior
            # and the prior.
            latent_stats = self.latent_model.sufficient_statistics(samples)
            prior_llh = self.latent_model.marginal_log_likelihood(latent_stats,
                                                                  **kwargs)
            kl_divs = post_llh - prior_llh

            decoder_means = self.decoder(samples)
            centered_data = (data - decoder_means)
            centered_stats = self.normal.sufficient_statistics(centered_data)
            llhs = self.normal.marginal_log_likelihood(centered_stats)

            # Store the statistics of the latent/likelihood model to
            # compute their gradients.
            self.cache['latent_stats'] = latent_stats.detach()
            self.cache['centered_stats'] = centered_stats.detach()
            s_llhs.append((llhs - kl_weight * kl_divs).view(-1, 1))
        return torch.cat(s_llhs, dim=-1).mean(dim=-1)

    def accumulate(self, _):
        latent_stats = self.cache['latent_stats']
        centered_stats = self.cache['centered_stats']
        return {
            **self.latent_model.accumulate(latent_stats),
            **self.normal.accumulate(centered_stats)
        }


class DualVAEGlobalMeanVariance(BayesianModel):
    '''Variational Auto-Encoder (VAE) with double latent space and
    a global mean and covariance matrix parameters.

    The second latent model/space is a "context" space (speaker space
    for instance).

    '''

    def __init__(self, encoder, encoder_problayer1, encoder_problayer2,
                 decoder, normal, latent_model1, latent_model2):
        super().__init__()
        self.encoder = encoder
        self.encoder_problayer1 = encoder_problayer1
        self.encoder_problayer2 = encoder_problayer2
        self.decoder = decoder
        self.normal = normal
        self.latent_model1 = latent_model1
        self.latent_model2 = latent_model2

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        #For the VAE, this is just the idenity function
        return data

    def mean_field_factorization(self):
        return self.latent_model1.mean_field_factorization() + \
            self.latent_model2.mean_field_factorization() + \
            self.normal.mean_field_factorization()

    def expected_log_likelihood(self, data, kl_weight=1., use_mean=False,
                                context_args={}, **kwargs):
        encoder_states = self.encoder(data)

        # Sample from the first latent space.
        posterior_params1 = self.encoder_problayer1(encoder_states)
        samples1, post_llh1 = self.encoder_problayer1.samples_and_llh(
            posterior_params1, use_mean)

        # Per-frame KL divergence between the (approximate) posterior
        # and the prior.
        latent_stats1 = self.latent_model1.sufficient_statistics(samples1)
        prior_llh1 = self.latent_model1.expected_log_likelihood(latent_stats1,
                                                                **kwargs)

        # Sample from the second latent space.
        sum_enc_states = encoder_states.mean(dim=0)[None, :]
        posterior_params2 = self.encoder_problayer2(sum_enc_states)
        samples2, post_llh2 = self.encoder_problayer2.samples_and_llh(
            posterior_params2, use_mean)

        # Per-frame KL divergence between the (approximate) posterior
        # and the prior.
        latent_stats2 = self.latent_model2.sufficient_statistics(samples2)
        prior_llh2 = self.latent_model2.expected_log_likelihood(latent_stats2,
                                                                **context_args)

        # Total KL divergence.
        kl_divs = post_llh1 - prior_llh1 + \
            (post_llh2 - prior_llh2) / len(encoder_states)

        # Since the second space is a "context space". It output only
        # a summary sample. We expand this summary vector to match the
        # other space number of samples.
        samples2 = samples2.view(-1).repeat(len(data), 1)

        samples = torch.cat([samples1, samples2], dim=-1)

        decoder_means = self.decoder(samples)
        centered_data = (data - decoder_means)
        centered_stats = self.normal.sufficient_statistics(centered_data)
        llhs = self.normal.expected_log_likelihood(centered_stats)

        # Store the statistics of the latent/likelihood model to
        # compute their gradients.
        self.cache['latent_stats1'] = latent_stats1.detach()
        self.cache['latent_stats2'] = latent_stats2.detach()
        self.cache['centered_stats'] = centered_stats.detach()
        return llhs - kl_weight * kl_divs

    def accumulate(self, _):
        latent_stats1 = self.cache['latent_stats1']
        latent_stats2 = self.cache['latent_stats2']
        centered_stats = self.cache['centered_stats']
        return {
            **self.latent_model1.accumulate(latent_stats1),
            **self.latent_model2.accumulate(latent_stats2),
            **self.normal.accumulate(centered_stats)
        }


__all__ = [
    'VAE',
    'VAEGlobalMeanVariance',
    'DualVAEGlobalMeanVariance'
]
