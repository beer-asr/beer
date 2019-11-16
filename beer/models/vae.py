
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import torch
from .basemodel import Model
from ..dists.normaldiag import NormalDiagonalCovariance

__all__ = ['VAE']
    
# Parameterization of the Normal using the
# log diagonal covariance matrix.
class MeanLogDiagCov(torch.nn.Module):

    def __init__(self, mean, log_diag_cov):
        super().__init__()
        self.mean = mean
        self.log_diag_cov = log_diag_cov

    @property
    def diag_cov(self):
        # Make sure the variance is never 0.
        return 1e-5 + self.log_diag_cov.exp()
        
        
class VAE(Model):
    
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.enc_mean_layer = torch.nn.Linear(encoder.dim_out, decoder.dim_in)
        self.enc_var_layer = torch.nn.Linear(encoder.dim_out, decoder.dim_in)
        self.dec_mean_layer = torch.nn.Linear(decoder.dim_out, encoder.dim_in)
        self.dec_var_layer = torch.nn.Linear(decoder.dim_out, encoder.dim_in)
         
    def posteriors(self, X):
        'Forward the data to the encoder to get the variational posteriors.'
        H = self.encoder(X)
        return NormalDiagonalCovariance(
            MeanLogDiagCov(self.enc_mean_layer(H), self.enc_var_layer(H))
        )
    
    def pdfs(self, Z):
        'Return the normal densities given the latent variable Z'
        Z1 = self.decoder(Z)
        return NormalDiagonalCovariance(
            MeanLogDiagCov(self.dec_mean_layer(Z1), self.dec_var_layer(Z1))
        )
    
    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        return self.prior.mean_field_factorization()

    def sufficient_statistics(self, data):
        return data

    def expected_log_likelihood(self, data, nsamples=1, llh_weight=1.,
                                kl_weight=1., **kwargs):
        posts = self.posteriors(data)
        
        # Local KL-divergence. There is a close for solution
        # for this term but we use sampling as it allows to
        # change the prior (GMM, HMM, ...) easily.
        samples = posts.sample(nsamples)
        s_samples = posts.sufficient_statistics(samples).mean(dim=1)
        ent = -posts(s_samples, pdfwise=True)
        s_samples = self.prior.sufficient_statistics(samples.view(-1, samples.shape[-1]))
        s_samples = s_samples.reshape(len(samples), -1, s_samples.shape[-1]).mean(dim=1)
        self.cache['prior_stats'] = s_samples
        xent = -self.prior.expected_log_likelihood(s_samples)
        local_kl_div = xent - ent
        
        # Approximate the expected log-likelihood with the
        # reparameterization trick.
        pdfs = self.pdfs(samples.view(-1, samples.shape[-1]))
        r_data = data[:, None, :].repeat(1, nsamples, 1).view(-1, data.shape[-1])
        llh = pdfs(pdfs.sufficient_statistics(r_data), pdfwise=True)
        llh = llh.reshape(len(data), nsamples, -1).mean(dim=1)
        
        return llh_weight * llh - kl_weight * local_kl_div

    def accumulate(self, stats, parent_msg=None):
        return self.prior.accumulate(self.cache['prior_stats'])