
"""Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

"""

import abc
import math

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from .model import Model
from ..priors import NormalGammaPrior, DirichletPrior
from ..training import mini_batches


class VAE(nn.Module):
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

    def evaluate(self, data, sampling=True):
        'Convenience function mostly for plotting and debugging.'
        torch_data = Variable(torch.from_numpy(data).float())
        state = self(torch_data, sampling=sampling)
        loss, llh, kld = self.loss(torch_data, state)
        return -loss, llh, kld, state['encoder_state'].mean, \
            state['encoder_state'].std_dev ** 2

    def forward(self, X, sampling=True):
        '''Forward data through the VAE model.

        Args:
            x (torch.Variable): Data to process. The first dimension is
                the number of samples and the second dimension is the
                dimension of the latent space.
            sampling (boolean): If True, sample to approximate the
                expectation of the log-likelihood.

        Returns:
            dict: State of the VAE.

        '''
        # Forward the data through the encoder.
        encoder_state = self.encoder(X)

        # Forward the statistics to the latent model.
        p_np_params, acc_stats = self.latent_model.expected_natural_params(
                encoder_state.mean.data,
                (1/encoder_state.prec.data))

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


class MLPEncoderState(metaclass=abc.ABCMeta):
    'Abstract Base Class for the state of a the VAE encoder.'

    @property
    def mean(self):
        'Mean of each distribution.'
        return self._mean

    @property
    def prec(self):
        'Diagonal of the precision matrix for each distribution.'
        return self._prec

    @abc.abstractmethod
    def sample(self):
        'sample data using the reparametization trick.'
        NotImplemented

    @abc.abstractmethod
    def kl_div(self, p_nparams):
        'kl divergence between the posterior and prior distribution.'
        NotImplemented


class MLPDecoderState(metaclass=abc.ABCMeta):
    'Abstract Base Class for the state of a the VAE decoder.'

    @abc.abstractmethod
    def natural_params(self):
        'Natural parameters for each distribution.'
        NotImplemented

    @abc.abstractmethod
    def log_base_measure(self, X):
        'Natural parameters for each distribution.'
        NotImplemented

    @abc.abstractmethod
    def sufficient_statistics(self, X):
        'Sufficient statistics of the given data.'
        NotImplemented

    def log_likelihood(self, X, nsamples=1):
        'Log-likelihood of the data.'
        s_stats = self.sufficient_statistics(X)
        nparams = self.natural_params()
        log_bmeasure = self.log_base_measure(X)
        nparams = nparams.view(nsamples, X.size(0), -1)
        return torch.sum(nparams * s_stats, dim=-1) + log_bmeasure


class MLPStateNormal(MLPEncoderState, MLPDecoderState):

    def __init__(self, mean, prec):
        self._mean = mean
        self._prec = prec

    def exp_T(self):
        idxs = torch.arange(0, self.mean.size(1)).long()
        XX = self.mean[:, :, None] * self.mean[:, None, :]
        XX[:, idxs, idxs] += 1 / self.prec
        return torch.cat([XX.view(self.mean.size(0), -1), self.mean,
                          Variable(torch.ones(self.mean.size(0), 2))], dim=-1)

    @property
    def std_dev(self):
        return 1 / torch.sqrt(self.prec)

    def natural_params(self):
        identity = Variable(torch.eye(self.mean.size(1)))
        np1 = -.5 * self.prec[:, None] * identity[None, :, :]
        np1 = np1.view(self.mean.size(0), -1)
        np2 = self.prec * self.mean
        np3 = -.5 * (self.prec * (self.mean ** 2)).sum(-1)[:, None]
        np4 = .5 * torch.log(self.prec).sum(-1)[:, None]
        return torch.cat([np1, np2, np3, np4], dim=-1)

    def sample(self):
        noise = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std_dev * noise

    def kl_div(self, p_nparams):
        return ((self.natural_params() - p_nparams) * self.exp_T()).sum(dim=-1)

    def sufficient_statistics(self, X):
        XX = X[:, :, None] * X[:, None, :]
        return torch.cat([XX.view(X.size(0), -1), X,
                          Variable(torch.ones(X.size(0), 2).float())], dim=-1)

    def log_base_measure(self, X):
        return -.5 * X.size(-1) * math.log(2 * math.pi)


class MLPStateNormalGamma(MLPDecoderState):

    def __init__(self, natural_params):
        self._natural_params = natural_params

    def natural_params(self):
        return self._natural_params

    def sufficient_statistics(self, X):
        return torch.cat([X, Variable(torch.ones(X.size(0), 1).float())],
                         dim=-1)

    def log_base_measure(self, X):
        return -.5 * X.size(-1) * math.log(2 * math.pi)

    def as_priors(self):
        '''Convert the current MLPState into a list of prior objects.

        Returns:
            list: The corresponding priors of the ``MLPState``.

        '''
        priors = [NormalGammaPrior(nparams.data.numpy()[:-1])
                  for nparams in self._natural_params]
        return priors


class MLPStateMixtureNormalGamma(MLPStateNormalGamma):

    @staticmethod
    def _mixture_priors_from_nparams(nparams, ncomps):
        np_comps = nparams[:-ncomps].reshape(ncomps, -1)
        components = [NormalGammaPrior(np_comp) for np_comp in np_comps]
        prior_weights = DirichletPrior(nparams[-ncomps:])
        return prior_weights, components

    def __init__(self, dim, ncomps, natural_params):
        super().__init__(natural_params)
        self._dim = dim
        self._ncomps = ncomps

    def sufficient_statistics(self, X):
        return torch.cat([X, Variable(torch.ones(X.size(0), 1).float())],
                         dim=-1)

    def log_base_measure(self, X):
        return self._ncomps * super().log_base_measure(X)

    def as_priors(self):
        priors = [self._mixture_priors_from_nparams(
                  nparams.data.numpy()[:-1], self._ncomps)
                  for nparams in self.natural_params()]
        return priors


class MLPModel(nn.Module):
    '''Base class for the encoder / decoder neural network of
    the VAE. The output of this network are the parameters of a
    conjugate exponential model. The proper way to use this class
    is to wrap with an object that "knows" how to make sense of the
    outptuts (see ``MLPEncoderState``, ``MLPDecoderIso``, ...).

    Note:
        This class only defines the neural network structure and does
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
        h = self.structure(X)
        outputs = [transform(h) for transform in self.output_layer]
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
        return MLPStateNormal(mean, torch.exp(logprec))


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
        return MLPStateNormal(mean,
            torch.exp(logprec) * Variable(torch.ones(mean.size(1)).float()))


class MLPNormalGamma(MLPModel):
    '''Neural-Network ending with 2 linear and non-linear projection
    corresponding to the parameters of the Normal-Gamma density. This
    MLP cannot be used as a decoder.

    '''

    @staticmethod
    def _get_natural_params(outputs, prior_count):
        # Get the standard parameters from the decoder.
        mean = next(outputs)
        # To avoid error we make sure that the variance is vanishing
        # by adding a small constant.
        var = 1e-2 + torch.log(1 + torch.exp(next(outputs)))
        a = prior_count * Variable(torch.ones(*mean.size()))
        b = prior_count * var

        # Convert the natural parameters to their natural form.
        np1 = prior_count * (mean ** 2) + 2 * b
        np2 = prior_count * mean
        np3 = prior_count * Variable(torch.ones(*mean.size()))
        np4 = 2 * a - 1

        # Commpute the log-normalizer w.r.t the natural parameters.
        lognorm = torch.lgamma(.5 * (np4 + 1))
        lognorm += -.5 * torch.log(np3)
        lognorm += -.5 * (np4 + 1) * torch.log(.5 * (np1 - ((np2**2)/ np3)))
        lognorm = lognorm.sum(dim=-1)

        return torch.cat([np1, np2, np3, np4, -lognorm[:, None]], dim=-1)

    def __init__(self, structure, dim, prior_count=1.):
        '''Initialize a ``MLPNormalGamma`` MLP.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            dim (int): Desired dimension of the modeled random
                variable.
            prior_count (float): Number of pseudo-observations.

        '''
        self.prior_count = prior_count
        super().__init__(structure, [(dim // 4, False)] * 2)

    def forward(self, X):
        outputs = iter(super().forward(X))
        return MLPStateNormalGamma(self._get_natural_params(outputs,
                                                            self.prior_count))


class MLPMixtureNormalGamma(MLPModel):
    '''Neural-Network outputing the conjugate prior of a Mixture model.'''

    def __init__(self, structure, dim, ncomps, prior_count=1.):
        '''Initialization.

        Args:
            structure (``torch.Sequential``): Sequence linear/
                non-linear operations.
            dim (int): Desired dimension of the modeled random
                variable.
            ncomps (int): number of components in the mixture.
            prior_count (float): Number of pseudo-observations.

        '''
        # Dimension of the MLP's output for 1 component of the mixture.
        dim_comp = (dim - ncomps) // (ncomps * 4)

        self.ncomps = ncomps
        self.prior_count = prior_count
        self.dim = dim
        final_projections = [(dim_comp, False)] * 2 * ncomps
        final_projections += [(ncomps, False)]
        super().__init__(structure, final_projections)

    def forward(self, X):
        outputs = iter(super().forward(X))
        lognorm = Variable(torch.zeros(X.size(0)))
        nparams = []
        for _ in range(self.ncomps):
            n = MLPNormalGamma._get_natural_params(outputs,
                self.prior_count)
            nparams.append(n[:, :-1])
            lognorm += -n[:, -1]

        # Prior over the weights.
        unnormed_w = 1 + torch.exp(next(outputs))
        w_nparams = self.prior_count * unnormed_w / torch.sum(unnormed_w, dim=-1)[:, None] - 1
        lognorm += -torch.lgamma(self.prior_count * Variable(torch.ones(w_nparams.size(0))))
        lognorm += torch.sum(torch.lgamma(w_nparams + 1), dim=-1)

        # Global natural parameters.
        nparams = torch.cat(nparams + [w_nparams, -lognorm[:, None]], dim=-1)

        return MLPStateMixtureNormalGamma(self.dim, self.ncomps, nparams)

