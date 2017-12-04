import numpy as np
import torch
from torch.autograd import Variable

from bokeh.plotting import figure, gridplot
from bokeh.io import show

import math


def run_training(training_data, model, optimizer, nb_epochs, history, batch_size,
                 lrate_latent_model):
    model.train()
    for epoch_no in range(1, nb_epochs+1):
        epoch_data = np.copy(training_data)
        np.random.shuffle(epoch_data)
        epoch_data_batched = epoch_data.reshape(-1, batch_size, 2)
        el = EpochLog()
        for X in epoch_data_batched:
            scale = float(epoch_data.shape[0]) / X.shape[0]
            X = Variable(torch.from_numpy(X).float())
            optimizer.zero_grad()
            state = model(X)
            loss, llh, kld = model.loss(X, state)
            loss.backward()
            optimizer.step()
            #model.latent_model.natural_grad_update(acc_stats,
            #    scale=scale, lrate=lrate_latent_model)

            el.accumulate(-loss.data[0], kld.data[0], llh.data[0])
            #el.accumulate(
            #    -loss.data[0], kld.data[0], llh.data[0],
            #    gaussian_entropy(z_logvar).data[0],
            #    gaussian_entropy(obs_logvar.view(-1, X.size(1))).data[0]
            #)

        history.log(el)


def gaussian_entropy(logvars):
    D = logvars.size(1)
    return 0.5 * (D + math.log(2*math.pi) + logvars.exp().sum()) / logvars.size(0)

class EpochLog:
    def __init__(self):
        self.elbo = 0
        self.kld = 0
        self.llh = 0
        self.h_z = 0
        self.h_x = 0

    def accumulate(self, elbo, kld, llh, h_z=None, h_x=None):
        self.elbo += elbo
        self.kld += kld
        self.llh += llh
        #self.h_z += h_z
        #self.h_x += h_x

    def __str__(self):
        return 'elbo: {:3.6f} llh: {:3.6f} kld: {:3.6f}'.format(
            self.elbo, self.llh, self.kld
        )
        #return "elbo: {:3.6f} llh: {:3.6f} kld: {:3.6f} H_q(z): {:3.6f} H_p(x|z): {:3.6f}".format(
        #            self.elbo, self.llh, self.kld, self.h_z, self.h_x
        #)


class History:
    def __init__(self, report_interval):
        self._epoch_logs = []
        self._report_interval = report_interval

    def log(self, epoch_log):
        self._epoch_logs.append(epoch_log)

        if len(self._epoch_logs) % self._report_interval == 0:
            print('Epoch: {} \t{}'.format(len(self._epoch_logs), epoch_log))

    def plot(self, subsampling=1):
        epoch_logs = self._epoch_logs[::subsampling]

        fig_elbo = figure(
            title='ELBO progress',
            width=400,
            height=400,
        )
        fig_elbo.line(range(len(epoch_logs)), [l.elbo for l in epoch_logs], color='blue')

        fig_kld = figure(
            title='KLD progress',
            width=400,
            height=400,
        )
        fig_kld.line(range(len(epoch_logs)), [l.kld for l in epoch_logs], color='red')

        fig_h_z = figure(
            title='H_q(z) progress',
            width=400,
            height=400,
        )
        fig_h_z.line(range(len(epoch_logs)), [l.h_z for l in epoch_logs], color='red')

        fig_h_x = figure(
            title='H_p(x|z) progress',
            width=400,
            height=400,
        )
        fig_h_x.line(range(len(epoch_logs)), [l.h_x for l in epoch_logs], color='red')

        gp = gridplot([
            [fig_elbo, fig_kld],
            [fig_h_z, fig_h_x],
        ])
        show(gp)

    def current_epoch(self):
        return len(self._epoch_logs)
