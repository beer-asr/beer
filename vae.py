import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


NB_SAMPLES = 10


class VAE(nn.Module):
    def __init__(self, latent_dim, nb_hidden):
        super(VAE, self).__init__()

        self.enc_in2hid = nn.Linear(2, nb_hidden)
        self.enc_hid2hid = nn.Linear(nb_hidden, nb_hidden)
        self.enc_hid2mu = nn.Linear(nb_hidden, latent_dim)
        self.enc_hid2var = nn.Linear(nb_hidden, latent_dim)

        self.dec_lat2hid = nn.Linear(latent_dim, nb_hidden)
        self.dec_hid2hid = nn.Linear(nb_hidden, nb_hidden)
        self.dec_hid2mu = nn.Linear(nb_hidden, 2)
        self.dec_hid2prec = nn.Linear(nb_hidden, 2)

        self.nonlin = F.relu

    def encode(self, x):
        h1 = self.nonlin(self.enc_in2hid(x))
        h2 = self.nonlin(self.enc_hid2hid(h1))
        return self.enc_hid2mu(h2), self.enc_hid2var(h2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h1 = self.nonlin(self.dec_lat2hid(z))
        h2 = self.nonlin(self.dec_hid2hid(h1))
        return self.dec_hid2mu(h2), self.dec_hid2prec(h2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        obs_mu, obs_logvar = self.decode(z)
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
