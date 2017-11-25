import math

import numpy as np
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib as mpl

NB_SAMPLES = 10

def plot_loss(model, fig,  orig_data, nb_samples=5, box=((-5, -5),(10.1, 10.1)), resolution=100):
    x_min = box[0][0]
    x_max = box[1][0]
    x_range = (x_max - x_min)
    x_res = x_range/resolution
    y_min = box[0][1]
    y_max = box[1][1]
    y_range = (y_max - y_min)
    y_res = y_range/resolution
    xy = np.mgrid[x_min:x_max:x_res, y_min:y_max:y_res].reshape(2,-1).T
    
    losses = []
    model.eval()
    for X in xy:
        X = Variable(torch.from_numpy(X).contiguous().float())
        X = torch.stack([X]*nb_samples)
        obs_mu, obs_logvar,  mu, logvar = model(X)
        loss, _, _ = loss_function(X, obs_mu, obs_logvar, mu, logvar)
        losses.append(math.exp(-loss.data.numpy()[0]))
    
    print(len(losses), type(losses[0]))
       
    losses_norm = mpl.colors.Normalize()(losses)
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), int(b)) 
        for r, g, b, _ in 255*plt.cm.Greys(losses_norm)
    ]
    
    fig.circle(xy[:, 0], xy[:, 1], fill_color=colors, line_color=colors, radius=min(x_range, y_range)/resolution/1.7)
    fig.cross(orig_data[:, 0], orig_data[:, 1], line_color='red')


def gauss_LLH(x, obs_mu, obs_logprec):
    LLH = (0.5*obs_logprec - 0.5*((x-obs_mu) * obs_logprec.exp() * (x-obs_mu))).sum()
    return LLH / NB_SAMPLES

def st_norm_KLD(z_mu, z_logvar, log_sigma=0.0):
    log_sigma = Variable(torch.FloatTensor([log_sigma]))
    KLD = 0.5 * torch.sum(log_sigma - 1 - z_logvar + z_logvar.exp()/log_sigma.exp() + z_mu.pow(2)/log_sigma.exp() )
    return KLD / NB_SAMPLES
    
def loss_function(x, obs_mu, obs_logprec, z_mu, z_logvar, log_sigma=0.0):
    LLH = gauss_LLH(x, obs_mu, obs_logprec)  
    KLD = st_norm_KLD(z_mu, z_logvar, log_sigma)

    return -(LLH - KLD), LLH, KLD


import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

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
        obs_mu, obs_logprec = self.decode(z)
        return obs_mu, obs_logprec, mu, logvar


def run_training(training_data, model, optimizer, nb_epochs, elbos, klds, batch_size, log_interval=10):
    model.train()
    for epoch_no in range(1, nb_epochs+1):
        epoch_data = np.copy(training_data)
        np.random.shuffle(epoch_data)
        epoch_data_batched = epoch_data.reshape(-1, batch_size, 2)
        epoch_elbo = 0
        epoch_kld = 0
        epoch_llh = 0
        for X in epoch_data_batched:
            X = Variable(torch.from_numpy(X).float())
            X = torch.cat([X] * NB_SAMPLES)
            optimizer.zero_grad()
            obs_mu, obs_logprec, z_mu, z_logvar = model(X)
            loss, llh, kld = loss_function(X, obs_mu, obs_logprec, z_mu, z_logvar, log_sigma=0.0)
            loss.backward()
            optimizer.step()

            epoch_elbo += -loss.data[0]
            epoch_kld += kld.data[0]
            epoch_llh += llh.data[0]
            
        elbos.append(epoch_elbo)
        klds.append(epoch_kld)    

        if epoch_no % log_interval == 0:
            print('Train Epoch: {} \telbo: {:.6f} \tllh: {:.6f}\tkld: {:.6f}'.format(
                epoch_no, epoch_elbo, epoch_llh, epoch_kld)
            )
