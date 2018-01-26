import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

def mini_batches(data, mini_batch_size, seed=None):
    rng = np.random.RandomState()
    if seed is not None:
        rng.seed(seed)
    indices = rng.choice(data.shape[0], size=data.shape[0], replace=False)
    splits = np.array_split(indices, data.shape[0] // mini_batch_size)
    for split in splits:
        yield data[split]


def train_vae(model, data, mini_batch_size=-1, max_epochs=1, seed=None, lrate=1e-3,
        latent_model_lrate=1., kl_weight=1.0, sample=True, callback=None):
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=1e-6)
    latent_model_lrate = latent_model_lrate
    data_size = np.prod(data.shape[:-1])

    mb_size = mini_batch_size if mini_batch_size > 0 else len(data)
    for epoch in range(1, max_epochs + 1):
        for mini_batch in mini_batches(data, mb_size, seed):
            # Forward the data through the VAE.
            X = Variable(torch.from_numpy(mini_batch).float())
            state = model(X, sample)

            nb_datapoints_in_batch = np.prod(mini_batch.shape[:-1])

            # Scale of the sufficient statistics.
            scale = float(data_size) / nb_datapoints_in_batch

            # Compute the loss (negative ELBO).
            loss, llh, kld = model.loss(X, state, kl_weight=kl_weight)
            loss, llh, kld = loss.sum(), llh.sum(), kld.sum()

            # We normalize the loss so we don't have to tune the learning rate
            # depending on the batch size.
            loss /= float(nb_datapoints_in_batch) # TODO -- PyTorch should take care of this

            # Neural updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Natural gradient step of the latent model.
            model.latent_model.natural_grad_update(state['acc_stats'],
                scale=scale, lrate=latent_model_lrate)

            # Full elbo (including the KL div. of the latent model).
            latent_model_kl = model.latent_model.kl_div_posterior_prior() / data_size
            lower_bound = -loss.data.numpy()[0] - latent_model_kl
            llh = llh.data.numpy()[0] / nb_datapoints_in_batch
            kld = kld.data.numpy()[0] / nb_datapoints_in_batch + latent_model_kl 

            if callback is not None:
                callback(lower_bound, llh, kld)

def train_conj_exp(model, data, mini_batch_size=-1, max_epochs=1, seed=None,
        lrate=1., callback=None):
    data_size= np.prod(data.shape[:-1])

    mb_size = mini_batch_size if mini_batch_size > 0 else len(data)
    for epoch in range(1, max_epochs + 1):
        for mini_batch in mini_batches(data, mb_size, seed):
            mini_batch_size = np.prod(mini_batch.shape[:-1])
            scale = float(data_size) / mini_batch_size

            exp_llhs, acc_stats = model.exp_llh(mini_batch, accumulate=True)
            exp_llh = np.sum(exp_llhs)
            kld = model.kl_div_posterior_prior()
            lower_bound = (scale * exp_llh - kld)
            model.natural_grad_update(acc_stats, scale, lrate)

            lower_bound = lower_bound / data_size
            llh = exp_llh / data_size
            kld = kld / data_size


            if callback is not None:
                callback(lower_bound, llh, kld)
