import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, latent_dim, nb_hidden):
        super(VAE, self).__init__()

        self.encoder = GaussianMLP(2, latent_dim, nb_hidden)
        self.decoder = GaussianMLP(latent_dim, 2, nb_hidden)
        self.latent_model = IsotropicGaussian(latent_dim)

        self.nb_samples = 10

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
        KLD = self.latent_model.kld(z_mu, z_logvar)
        return -(LLH - KLD), LLH, KLD

    def gauss_LLH(self, x, obs_mu, obs_logvar):
        LLH = 0.5 * (
            (-obs_logvar) +
            (- (x-obs_mu).pow(2) * (-obs_logvar).exp())
        ).sum()
        return LLH / obs_mu.size(0)


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
