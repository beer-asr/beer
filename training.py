import numpy as np
import torch
from torch.autograd import Variable

from bokeh.plotting import figure, gridplot
from bokeh.io import show


def run_training(training_data, model, optimizer, nb_epochs, history, batch_size, log_interval=10):
    model.train()
    for epoch_no in range(1, nb_epochs+1):
        epoch_data = np.copy(training_data)
        np.random.shuffle(epoch_data)
        epoch_data_batched = epoch_data.reshape(-1, batch_size, 2)
        el = EpochLog()
        for X in epoch_data_batched:
            X = Variable(torch.from_numpy(X).float())
            optimizer.zero_grad()
            obs_mu, obs_logvar, z_mu, z_logvar = model(X)
            loss, llh, kld = model.loss(X, obs_mu, obs_logvar, z_mu, z_logvar)
            loss.backward()
            optimizer.step()

            el.accumulate(-loss.data[0], kld.data[0], llh.data[0])

        history.log(el)

        if epoch_no % log_interval == 0:
            print('Train Epoch: {} \t{}'.format(epoch_no, el))

class EpochLog:
    def __init__(self):
        self.elbo = 0
        self.kld = 0
        self.llh = 0

    def accumulate(self, elbo, kld, llh):
        self.elbo += elbo
        self.kld += kld
        self.llh += llh

    def __str__(self):
        return "elbo: {} llh: {} kld: {}".format(self.elbo, self.llh, self.kld)


class History:
    def __init__(self):
        self._epoch_logs = []

    def log(self, epoch_log):
        self._epoch_logs.append(epoch_log)

    def plot(self, subsampling=1):
        epoch_logs = self._epoch_logs[::subsampling]

        fig_elbo = figure(
            title='ELBO Progress',
            width=400,
            height=400,
        )
        fig_elbo.line(range(len(epoch_logs)), [l.elbo for l in epoch_logs], color='blue')

        fig_kld = figure(
            title='KLD Progress',
            width=400,
            height=400,
        )
        fig_kld.line(range(len(epoch_logs)), [l.kld for l in epoch_logs], color='red')

        gp = gridplot([[fig_elbo, fig_kld]])
        show(gp)

    def expected_epoch(self):
        return len(self._epoch_logs)
