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
        epoch_elbo = 0
        epoch_kld = 0
        epoch_llh = 0
        for X in epoch_data_batched:
            X = Variable(torch.from_numpy(X).float())
            optimizer.zero_grad()
            obs_mu, obs_logvar, z_mu, z_logvar = model(X)
            loss, llh, kld = model.loss(X, obs_mu, obs_logvar, z_mu, z_logvar)
            loss.backward()
            optimizer.step()

            epoch_elbo += -loss.data[0]
            epoch_kld += kld.data[0]
            epoch_llh += llh.data[0]

        history.log(epoch_elbo, epoch_kld)

        if epoch_no % log_interval == 0:
            print('Train Epoch: {} \telbo: {:.6f} \tllh: {:.6f}\tkld: {:.6f}'.format(
                epoch_no, epoch_elbo, epoch_llh, epoch_kld)
            )


class History:
    def __init__(self):
        self._elbos = []
        self._klds = []

    def log(self, elbo, kld):
        self._elbos.append(elbo)
        self._klds.append(kld)

    def plot(self, subsampling=1):
        fig_elbo = figure(
            title='ELBO Progress',
            width=400,
            height=400,
        )
        elbos_subsampled = self._elbos[::subsampling]
        fig_elbo.line(range(len(elbos_subsampled)), elbos_subsampled, color='blue')

        fig_kld = figure(
            title='KLD Progress',
            width=400,
            height=400,
        )
        klds_subsampled = self._klds[::subsampling]
        fig_kld.line(range(len(klds_subsampled)), klds_subsampled, color='red')

        gp = gridplot([[fig_elbo, fig_kld]])
        show(gp)
