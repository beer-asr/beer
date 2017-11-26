import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


NB_SAMPLES = 10


class VAE(nn.Module):
    def __init__(self, latent_dim, nb_hidden):
        super(VAE, self).__init__()

        self.encoder = GaussianMLP(2, latent_dim, nb_hidden)
        self.decoder = GaussianMLP(latent_dim, 2, nb_hidden)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        obs_mu, obs_logvar = self.decoder(z)
        return obs_mu, obs_logvar, mu, logvar

    def loss(self, x, obs_mu, obs_logvar, z_mu, z_logvar, log_sigma=0.0):
        LLH = gauss_LLH(x, obs_mu, obs_logvar)
        KLD = st_norm_KLD(z_mu, z_logvar, log_sigma)
        return -(LLH - KLD), LLH, KLD


def gauss_LLH(x, obs_mu, obs_logvar):
    LLH = 0.5 * (
        (-obs_logvar) +
        (- (x-obs_mu) * (-obs_logvar).exp() * (x-obs_mu))
    ).sum()
    return LLH / NB_SAMPLES


def st_norm_KLD(z_mu, z_logvar, log_sigma=0.0):
    log_sigma = Variable(torch.FloatTensor([log_sigma]))
    KLD = 0.5 * torch.sum(
        (log_sigma - 1) +
        (- z_logvar) +
        z_logvar.exp()/log_sigma.exp() +
        z_mu.pow(2)/log_sigma.exp()
    )
    return KLD / NB_SAMPLES


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, nb_hidden):
        super(GaussianMLP, self).__init__()

        self.in2hid = nn.Linear(input_dim, nb_hidden)
        self.hid2hid = nn.Linear(nb_hidden, nb_hidden)
        self.hid2mu = nn.Linear(nb_hidden, output_dim)
        self.hid2logvar = nn.Linear(nb_hidden, output_dim)

        self.nonlin = F.relu

    def forward(self, x):
        h1 = self.nonlin(self.in2hid(x))
        h2 = self.nonlin(self.hid2hid(h1))
        return self.hid2mu(h2), self.hid2logvar(h2)
