
'Discriminative Variational Model.'

import abc
import math

import torch
from torch import nn
from torch.autograd import Variable


def _log_softmax(inputs):
    s, _ = torch.max(inputs, dim=1, keepdim=True)
    norm = s + (inputs - s).exp().sum(dim=1, keepdim=True).log()
    return inputs - norm.view(-1, 1)


class DiscriminativeVariationalModel(nn.Module):
    'Discriminative Variational Model (DVM).'

    def __init__(self, encoder, latent_model, nsamples=10):
        '''Initialize the DVM.

        Args:
            encoder (``MLPModel``): Encoder neural network..
            latent_model(``Mixture``): Bayesian Model
                for the prior over the latent space.

        '''
        super().__init__()
        self.encoder = encoder
        self.latent_model = latent_model
        self._xent_loss_fn = nn.CrossEntropyLoss()
        self.nsamples = nsamples

    def evaluate(self, data, labels):
        'Convenience function mostly for plotting and debugging.'
        state = self(data, sampling=False)
        loss, llh, kld = self.loss(data, labels, state)
        return -loss, llh, kld, state['encoder_state'].mean, \
            state['encoder_state'].std_dev ** 2

    def forward(self, X, labels=None, sampling=True):
        '''Forward data through the DVM.

        Args:
            X (Variable): Data to process. The first dimension is
                the number of samples and the second dimension is the
                dimension of the latent space.

        Returns:
            ``MLPEncoderState``: State of the DVM.

        '''
        state = self.encoder(X)
        mean, var = state.mean, (1. / state.prec)

        # Samples of the latent variable using the reparameterization
        # "trick". "z" is a L x N x K tensor where L is the number of
        # samples for the reparameterization "trick", N is the number
        # of frames and K is the dimension of the latent space.
        if sampling:
            nsamples = self.nsamples
            samples = []
            for i in range(self.nsamples):
                samples.append(state.sample())
            samples = torch.stack(samples).view(self.nsamples * X.size(0), -1)
        else:
            nsamples = 1
            samples = mean

        # Pre-softmax layer.
        T = self.latent_model.sufficient_statistics(samples)
        weights_and_bias = self.latent_model.expected_comp_params
        presoftmax = T @ Variable(weights_and_bias.t())

        # Forward the statistics to the latent model.
        p_np_params, _ = self.latent_model.expected_natural_params(
                state.mean.data, var.data)

        retval = {
            'nsamples': nsamples,
            'encoder_state': state,
            'p_np_params': Variable(torch.FloatTensor(p_np_params)),
            'presoftmax': presoftmax,
        }

        if labels is not None:
            T = self.latent_model.sufficient_statistics_from_mean_var(mean, var)
            elabels = self.latent_model._expand_labels(labels.data,
                len(self.latent_model.components))
            acc_stats = elabels.t() @ T.data[:, :-1], elabels.sum(dim=0)
            retval['acc_stats'] = acc_stats

        return retval

    def predictions(self, X, sampling=False):
        state = self(X, sampling=sampling)
        nsamples = state['nsamples']
        llh =  _log_softmax(state['presoftmax'])
        llh = llh.view(nsamples, X.size(0), -1).sum(dim=0)
        return llh

    def loss(self, X, labels, state, kl_weight=1.0):
        '''Loss function of the DVM. This is the negative of the
        variational objective function i.e.:

            loss = - ( E_q [ ln p(Z|X) ] - KL( q(X) || p(X) ) )

        Args:
            X (torch.Variable): Data on which to estimate the loss.
            labels (torch.Variable): Labels for each frame.
            state (dict): Current state of the CVM.
            kl_weight (float): Weight of the KL divergence in the loss.
                You probably don't want to touch it unless you know
                what you are doing.

        Returns:
            torch.Variable: Symbolic computation of the loss function.

        '''
        nsamples = state['nsamples']
        elabels = self.latent_model._expand_labels(labels.data,
            len(self.latent_model.components))
        kl = state['encoder_state'].kl_div(state['p_np_params']) * kl_weight
        llh =  _log_softmax(state['presoftmax'])
        llh = llh.view(nsamples, X.size(0), -1).sum(dim=0)
        llh = (Variable(elabels) * llh).sum(dim=-1)
        return -(llh - kl), llh, kl

