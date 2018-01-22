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
            # Number of samples in the mini-batch
            mini_batch_size = np.prod(mini_batch.shape[:-1])

            # Scale of the sufficient statistics.
            scale = float(data_size) / mini_batch_size

            # Convert the data into the suitable pytorch Variable
            X = Variable(torch.from_numpy(mini_batch).float())

            # Clean up the previously accumulated gradient.
            optimizer.zero_grad()

            # Forward the data through the VAE.
            state = model(X, sample)

            # Compute the loss (negative ELBO).
            loss, llh, kld = model.loss(X, state, kl_weight=kl_weight)
            loss, llh, kld = loss.sum(), llh.sum(), kld.sum()

            # We normalize the loss so we don't have to tune the learning rate
            # depending on the batch size.
            loss /= float(mini_batch_size)

            # Backward propagation of the gradient.
            loss.backward()

            # Update of the parameters of the neural network part of the
            # model.
            optimizer.step()

            # Natural gradient step of the latent model.
            model.latent_model.natural_grad_update(state['acc_stats'],
                scale=scale, lrate=latent_model_lrate)

            # Full elbo (including the KL div. of the latent model).
            latent_model_kl = model.latent_model.kl_div_posterior_prior() / data_size
            elbo = -loss.data.numpy()[0] - latent_model_kl

            lower_bound = elbo
            llh = llh.data.numpy()[0] / mini_batch_size
            kld = kld.data.numpy()[0] / mini_batch_size + latent_model_kl 

            if callback is not None:
                callback(lower_bound, llh, kld)
