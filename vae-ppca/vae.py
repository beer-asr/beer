
"""Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_model, nb_samples):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nb_samples = nb_samples

    def reparameterize(self, mu, logvar):
        if self.training:
            z_samples = []
            for i in range(self.nb_samples):
                std = logvar.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                z_samples.append(eps.mul(std).add_(mu))

            return torch.stack(z_samples)
        else:
            return mu.unsqueeze(0)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        obs_mu, obs_logvar = self.decoder(z.view(-1, 2))

        obs_mu = obs_mu.view(-1, x.size(0), 2)
        obs_logvar = obs_logvar.view(-1, x.size(0), 2)
        return obs_mu, obs_logvar, mu, logvar

    def loss(self, x, obs_mu, obs_logvar, z_mu, z_logvar):
        LLH = self.gauss_LLH(x, obs_mu, obs_logvar)
        KLD = self.latent_model.kld(*naturals_from_mu_logvar(z_mu, z_logvar))
        return -(LLH - KLD), LLH, KLD

    def gauss_LLH(self, x, obs_mu, obs_logvar):
        LLH = 0.5 * (
            (-obs_logvar) +
            (- (x-obs_mu).pow(2) * (-obs_logvar).exp())
        ).sum()
        return LLH / obs_mu.size(0)


class GaussianMLP(nn.Module):
    """ Neural-Network ending with a double linear projectiong 
    providing the mean and the (log-)variance a of a Normal 
    distribution given the input.

    """
    
    def __init__(self, structure, hidden_dim, target_dim):
        super().__init__()
        self.structure = structure
        self.hid_to_mu = nn.Linear(hidden_dim, target_dim)
        self.hid_to_logvar = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        h = self.structure(x)
        return self.hid_to_mu(h), self.hid_to_logvar(h)
        

class IsotropicGaussian:
    def __init__(self, dim, log_var=0.0):
        self._dim = dim
        self._log_var = Variable(torch.FloatTensor([log_var]))

    def kld(self, mu, logvar):
        assert(mu.size(1) == self._dim)

        KLD = 0.5 * torch.sum(
            (self._log_var - 1) +
            (- logvar) +
            logvar.exp()/self._log_var.exp() +
            mu.pow(2)/self._log_var.exp()
        )
        return KLD

    def update(self, mu, logvar):
        pass


class NaturalIsotropicGaussian:
    def __init__(self, dim, log_var=0.0):
        var = math.exp(log_var)
        self._np_linear = Variable(torch.FloatTensor([0]*dim))
        self._np_quadr = Variable(torch.FloatTensor([-0.5*var]*dim))

    def kld(self, q_np_linear, q_np_quadr, stats_linear, stats_quadr):
        KLD = ((q_np_linear - self._np_linear) * stats_linear).sum(dim=1)
        KLD += ((q_np_quadr - self._np_quadr) * stats_quadr).sum(dim=1)
        KLD += log_norm(self._np_linear.view(1, -1), self._np_quadr.view(1,-1))
        KLD += -log_norm(q_np_linear, q_np_quadr)
        KLD = KLD.sum()

        return KLD

    def update(self, mu, logvar):
        pass


def log_norm(np_linear, np_quadr):
    # print(np_linear)
    # print(np_quadr)
    return - 0.5 * torch.log((-2*np_quadr)).sum(dim=1) - 0.25 * (np_linear.pow(2) * (1.0/np_quadr)).sum(dim=1)


def naturals_from_mu_logvar(mu, logvar):
    var = logvar.exp()
    stats_linear = mu
    stats_quadr = mu.pow(2) + var

    np_linear = mu / var
    np_quadr = -0.5 / var
    return np_linear, np_quadr, stats_linear, stats_quadr
