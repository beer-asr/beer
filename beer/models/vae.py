
"""Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

"""

import abc
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np
from .model import Model


class VAE(nn.Module, Model):
    """Variational Auto-Encoder (VAE)."""

    def __init__(self, encoder, decoder, latent_model, nsamples):
        """Initialize the VAE.

        Args:
            encoder (``MLPModel``): Encoder of the VAE.
            decoder (``MLPModel``): Decoder of the VAE.
            latent_model(``ConjugateExponentialModel``): Bayesian Model
                for the prior over the latent space.
            nsamples (int): Number of samples to approximate the
                expectation of the log-likelihood.

        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.nsamples = nsamples

    def _fit_step(self, mini_batch):
        # Number of samples in the mini-batch
        mini_batch_size = np.prod(mini_batch.shape[:-1])

        # Total number of samples of the training data.
        data_size = self._fit_cache['data_size']

        # Scale of the sufficient statistics.
        scale = float(data_size) / mini_batch_size

        # Convert the data into the suitable pytorch Variable
        X = Variable(torch.from_numpy(mini_batch).float())

        # Clean up the previously accumulated gradient.
        self._fit_cache['optimizer'].zero_grad()

        # Forward the data through the VAE.
        state = self(X)

        # Compute the loss (negative ELBO).
        loss, llh, kld = self.loss(X, state,
                                   kl_weight=self._fit_cache['kl_weight'])
        loss, llh, kld = loss.sum(), llh.sum(), kld.sum()

        # We normalize the loss so we don't have to tune the learning rate
        # depending on the batch size.
        loss /= float(mini_batch_size)

        # Backward propagation of the gradient.
        loss.backward()

        # Update of the parameters of the neural network part of the
        # model.
        self._fit_cache['optimizer'].step()

        # Natural gradient step of the latent model.
        self.latent_model.natural_grad_update(state['acc_stats'],
            scale=scale, lrate=self._fit_cache['latent_model_lrate'])

        # Full elbo (including the KL div. of the latent model).
        latent_model_kl = self.latent_model.kl_div_posterior_prior() / data_size
        elbo = -loss.data.numpy()[0] - latent_model_kl

        return elbo, llh.data.numpy()[0] / mini_batch_size, \
            kld.data.numpy()[0] / mini_batch_size + latent_model_kl

    def fit(self, data, mini_batch_size=-1, max_epochs=1, seed=None, lrate=1e-3,
            latent_model_lrate=1., kl_weight=1.0, callback=None):
        self._fit_cache = {
            'optimizer':optim.Adam(self.parameters(), lr=lrate,
                                   weight_decay=1e-6),
            'latent_model_lrate': latent_model_lrate,
            'data_size': np.prod(data.shape[:-1]),
            'kl_weight': kl_weight
        }
        super().fit(data, mini_batch_size, max_epochs, seed, callback)

    def evaluate(self, data, sampling=True):
        'Convenience function mostly for plotting and debugging.'
        torch_data = Variable(torch.from_numpy(data).float())
        state = self(torch_data, sampling=sampling)
        loss, llh, kld = self.loss(torch_data, state)
        return -loss.data.numpy(), llh.data.numpy(), kld.data.numpy(), \
            state['encoder_state'].mean.data.numpy(), \
            state['encoder_state'].std_dev().data.numpy()**2, \
            state['decoder_state'].mean.data.numpy(), \
            state['decoder_state'].std_dev().data.numpy()**2


    def forward(self, X, sampling=True):
        '''Forward data through the VAE model.

        Args:
            x (torch.Variable): Data to process. The first dimension is
                the number of samples and the second dimension is the
                dimension of the latent space.
            sampling (boolearn): If True, sample to approximate the
                expectation of the log-likelihood.

        Returns:
            dict: State of the VAE.

        '''
        # Forward the data through the encoder.
        encoder_state = self.encoder(X)

        # Forward the statistics to the latent model.
        p_np_params, acc_stats = self.latent_model.expected_natural_params(
                encoder_state.mean.data.numpy(),
                (1/encoder_state.prec).data.numpy())

        # Samples of the latent variable using the reparameterization
        # "trick". "z" is a L x N x K tensor where L is the number of
        # samples for the reparameterization "trick", N is the number
        # of frames and K is the dimension of the latent space.
        if sampling:
            nsamples = self.nsamples
            samples = []
            for i in range(self.nsamples):
                samples.append(encoder_state.sample())
            samples = torch.stack(samples).view(self.nsamples * X.size(0), -1)
            decoder_state = self.decoder(samples)
        else:
            nsamples = 1
            decoder_state = self.decoder(encoder_state.mean)

        return {
            'encoder_state': encoder_state,
            'p_np_params': Variable(torch.FloatTensor(p_np_params)),
            'acc_stats': acc_stats,
            'decoder_state': decoder_state,
            'nsamples': nsamples
        }

    def loss(self, X, state, kl_weight=1.0):
        """Loss function of the VAE. This is the negative of the
        variational objective function i.e.:

            loss = - ( E_q [ ln p(X|Z) ] - KL( q(z) || p(z) ) )

        Args:
            X (torch.Variable): Data on which to estimate the loss.
            state (dict): State of the VAE after forwarding the data
                through the network.
            kl_weight (float): Weight of the KL divergence in the loss.
                You probably don't want to touch it unless you know
                what you are doing.

        Returns:
            torch.Variable: Symbolic computation of the loss function.

        """
        nsamples = state['nsamples']
        llh = state['decoder_state'].log_likelihood(X, state['nsamples'])
        llh = llh.view(nsamples, X.size(0), -1).sum(dim=0) / nsamples
        kl = state['encoder_state'].kl_div(state['p_np_params'])
        kl *= kl_weight

        return -(llh - kl[:, None]), llh, kl


class NormalPosteriors:
    '''Object that wraps the output of a MLP and behaves as a set of
    Normal distribution (with diagional covariance matrix).

    '''

    def __init__(self, mean, prec):
        '''Initialize the set of Normal distribution.

        Args:
            mean (torch.autograd.Variable): Mean for each Normal
                distribution.
            logprec (torch.autograd.Variable): Log-precision for each
                Normal distribution.

        '''
        self.mean = mean
        self.prec = prec

    def exp_T(self):
        'Expected sufficient statistics for each distribution.'
        idxs = torch.arange(0, self.mean.size(1)).long()
        XX = self.mean[:, :, None] * self.mean[:, None, :]
        XX[:, idxs, idxs] += 1 / self.prec
        return torch.cat([XX.view(self.mean.size(0), -1), self.mean,
                          Variable(torch.ones(self.mean.size(0), 2))], dim=-1)

    def std_dev(self):
        'Standard deviation (per-dimension) for each distribution.'
        return 1 / torch.sqrt(self.prec)

    def natural_params(self):
        'Natural parameters for each of the distribution.'
        identity = Variable(torch.eye(self.mean.size(1)))
        np1 = -.5 * self.prec[:, None] * identity[None, :, :]
        np1 = np1.view(self.mean.size(0), -1)
        np2 = self.prec * self.mean
        np3 = -.5 * (self.prec * (self.mean ** 2)).sum(-1)[:, None]
        np4 = .5 * torch.log(self.prec).sum(-1)[:, None]
        return torch.cat([np1, np2, np3, np4], dim=-1)

    @staticmethod
    def sufficient_statistics(X):
        'Sufficient statistics of the given data.'
        XX = X[:, :, None] * X[:, None, :]
        return torch.cat([XX.view(X.size(0), -1), X,
                          Variable(torch.ones(X.size(0), 2).float())], dim=-1)

    def log_likelihood(self, X, nsamples=1):
        'Log-likelihood of the data.'
        s_stats = self.sufficient_statistics(X)
        nparams = self.natural_params()
        nparams = nparams.view(nsamples, X.size(0), -1)
        log_base_measure = -.5 * X.size(-1) * math.log(2 * math.pi)
        return torch.sum(nparams * s_stats, dim=-1) + log_base_measure

    def sample(self):
        '''sample data using the reparametization trick.'''
        noise = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std_dev() * noise

    def kl_div(self, p_nparams):
        '''kl divergence between the posterior distribution and the
        prior.

        '''
        return ((self.natural_params() - p_nparams) * self.exp_T()).sum(dim=-1)


class MLPModel(nn.Module):
    '''Base class for the encoder / decoder neural network of
    the VAE. The output of this network are the parameters of a
    conjugate exponential model. The proper way to use this class
    is to wrap with an object that "knows" how to make sense of the
    outptuts (see ``MLPNormalDiag``, ``MLPNormalIso``, ...).

    Note:
        This class only define the neural network structure and does
        not care wether it is used as encoder/decoder and how the
        parameters of the model is used.

    '''

    @staticmethod
    def _init_residulal_layer(linear_transform):
        W = linear_transform.weight.data.numpy()
        dim = max(*W.shape)
        q, _ = np.linalg.qr(np.random.randn(dim, dim))
        W = q[:W.shape[0], :W.shape[1]]
        linear_transform.weight = nn.Parameter(torch.from_numpy(W).float())

    def __init__(self, structure, outputs):
        '''Initialize the ``MLPModel``.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            outputs (list): List of tuple describing the output model.

        '''
        super().__init__()
        self.structure = structure

        # Get the input/ouput dimension of the structure.
        for transform in structure:
            if isinstance(transform, nn.Linear):
                in_dim = transform.in_features
                break
        for transform in reversed(structure):
            if isinstance(transform, nn.Linear):
                out_dim = transform.out_features
                break

        # Create the specific output layer.
        self.output_layer = nn.ModuleList()
        self.residual_connections = nn.ModuleList()
        self.residual_mapping = {}
        for i, output in enumerate(outputs):
            target_dim, residual = output
            self.output_layer.append(nn.Linear(out_dim, target_dim))
            if residual:
                ltransform = nn.Linear(in_dim, target_dim)
                MLPModel._init_residulal_layer(ltransform)
                self.residual_connections.append(ltransform)
                self.residual_mapping[i] = len(self.residual_connections) - 1

    def forward(self, X):
        # Forward the data through the inner structure of the model.
        h = self.structure(X)

        # Get the final outputs:
        outputs = [transform(h) for transform in self.output_layer]

        # Apply the residual connection (if any).
        for idx1, idx2 in self.residual_mapping.items():
            outputs[idx1] += self.residual_connections[idx2](X)

        return outputs

class MLPNormalDiag(MLPModel):
    '''Neural-Network ending with a double linear projection
    providing the mean and the logarithm of the diagonal of the
    covariance matrix.

    '''

    def __init__(self, structure, dim, residual=False):
        '''Initialize a ``MLPNormalDiag`` object.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            dim (int): Desired dimension of the modeled random
                variable.
            residual (boolean): Add a residual connection to the mean.

        '''
        super().__init__(structure, [(dim, residual), (dim, False)])

    def forward(self, X):
        mean, logprec = super().forward(X)
        return NormalPosteriors(mean, torch.exp(logprec))


class MLPNormalIso(MLPModel):
    '''Neural-Network ending with a double linear projection
    providing the mean and the isotropic covariance matrix.

    '''

    def __init__(self, structure, dim, residual=False):
        '''Initialize a ``MLPNormalDiag`` object.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            dim (int): Desired dimension of the modeled random
                variable.
            residual (boolean): Add a residual connection to the mean.

        '''
        super().__init__(structure, [(dim, residual), (1, False)])

    def forward(self, X):
        mean, logprec = super().forward(X)
        return NormalPosteriors(mean,
            torch.exp(logprec) * Variable(torch.ones(mean.size(1)).float()))

