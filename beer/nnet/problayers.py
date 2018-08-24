
import abc
import math
import torch


class ProbabilisticLayer(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Probabilistic layer to be used by the encoder/decoder of a
    Variational AutoEncoder.

    '''

    @abc.abstractmethod
    def forward(self, inputs):
        '''Compute the parameters of the distribution conditioned on the
        input.

        Args:
            inputs (``torch.Tensor[N,dim]``): Conditional inputs.

        Returns:
            params (object): Parameters of the distribution.

        '''
        pass

    @abc.abstractmethod
    def samples_and_llh(self, params, use_mean=False):
        '''Samples using the reparameterization trick so that the each
        sample can be backpropagated trough. In addition, returns the
        log-likelihood of the samples given the sampling distribution.

        Args:
            params (object): Parameters of the sampling distribution.
            use_mean (boolean): If true, by pass the sampling and
                just return the mean value of the distribution.

        Returns:
            samples (``torch.Tensor``): Sampled values.
            llh (``torch.Tensor``): Log-likelihood for each sample.
        '''
        pass

    @abc.abstractmethod
    def log_likelihood(self, data, params):
        '''Log-likelihood of the data.

        Args:
            data (``torch.Tensor[N,dim]``): Data.
            params (object): Parameters of the distribution.

        '''
        pass


class NormalDiagonalCovarianceLayer(ProbabilisticLayer):
    'Normal distribution with diagonal covariance matrix layer.'

    def __init__(self, dim_in, dim_out, variance_nonlinearity=None):
        super().__init__()
        self.mean = torch.nn.Linear(dim_in, dim_out)
        self.logvar = torch.nn.Linear(dim_in, dim_out)
        if variance_nonlinearity is None:
            variance_nonlinearity = torch.nn.Softplus()
        self.variance_nonlinearity = variance_nonlinearity

    def forward(self, inputs):
        return self.mean(inputs), \
               self.variance_nonlinearity(self.logvar(inputs))

    def samples_and_llh(self, params, use_mean=False):
        means, variances = params
        if use_mean:
            samples = means
        else:
            dtype, device = means.dtype, means.device
            noise = torch.randn(*means.shape, dtype=dtype, device=device)
            std_dev = variances.sqrt()
            samples = means + std_dev * noise
        llhs = self.log_likelihood(samples, params)
        return samples, llhs

    def log_likelihood(self, data, params):
        means, variances = params
        dim = means.shape[-1]
        delta = torch.sum((data - means).pow(2) / variances, dim=-1)
        return -.5 * (variances.log().sum(dim=-1) + delta + dim * math.log(2 * math.pi))


class NormalIsotropicCovarianceLayer(NormalDiagonalCovarianceLayer):
    'Normal distribution with isotropic covariance matrix layer.'

    def __init__(self, dim_in, dim_out,
                variance_nonlinearity=None):
        super().__init__(dim_in, dim_out)
        self.mean = torch.nn.Linear(dim_in, dim_out)
        self.logvar = torch.nn.Linear(dim_in, 1)
        if variance_nonlinearity is None:
            variance_nonlinearity = torch.nn.Softplus()
        self.variance_nonlinearity = variance_nonlinearity

    def forward(self, inputs):
        means = self.mean(inputs)
        padding = torch.ones_like(means)
        variances = self.variance_nonlinearity(self.logvar(inputs)) * padding
        return means, variances


class NormalIdentityCovarianceLayer(ProbabilisticLayer):
    'Normal distribution with identity covariance matrix layer.'

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mean = torch.nn.Linear(dim_in, dim_out)

    def forward(self, inputs):
        return self.mean(inputs)

    def samples_and_llh(self, params, use_mean=False):
        means = params
        if use_mean:
            samples = means
        else:
            dtype, device = means.dtype, means.device
            noise = torch.randn(*means.shape, dtype=dtype, device=device)
            samples = means + noise
        llhs = self.log_likelihood(samples, params)
        return samples, llhs

    def log_likelihood(self, data, params):
        means = params
        dim = means.shape[-1]
        delta = torch.sum((data - means).pow(2), dim=-1)
        return -.5 * (delta + dim * math.log(2 * math.pi))


class BernoulliLayer(ProbabilisticLayer):
    'Bernoulli distribution layer.'

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mean = torch.nn.Linear(dim_in, dim_out)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        return self.sigmoid(self.mean(inputs))

    def samples_and_llh(self, params, use_mean=False):
        '''The Bernoulli layer cannot be used as an encoding
        distribution since we cannot backpropagate through discrete
        samples.

        '''
        raise NotImplementedError

    def log_likelihood(self, data, params):
        means = params
        epsilon = 1e-6
        per_pixel_bce = data * torch.log(epsilon + means) + \
            (1.0 - data) * torch.log(epsilon + 1 - means)
        return per_pixel_bce.sum(dim=-1)


class InverseAutoRegressiveFlow(ProbabilisticLayer):
    'Inverse auto-regressive flow.'


    def __init__(self, dim_in, flow_params_dim, normal_layer, nnet_flow):
        super().__init__()
        self.normal_layer = normal_layer
        self.flow_params = torch.nn.Linear(dim_in, flow_params_dim)
        self.nnet_flow = nnet_flow

    def forward(self, inputs):
        flow_params = self.flow_params(inputs)
        normal_params = self.normal_layer(inputs)
        return (normal_params, flow_params)

    def samples_and_llh(self, params, use_mean=False):
        normal_params, flow_params = params
        means, variances = normal_params
        dim = means.shape[-1]
        if use_mean:
            flow = means
            llhs = torch.zeros_like(means[:, 0])
        else:
            dtype, device = means.dtype, means.device
            noise = torch.randn(*means.shape, dtype=dtype, device=device)
            std_dev = variances.sqrt()
            flow = means + std_dev * noise
            llhs = -.5 * ((noise ** 2).sum(dim=-1) + dim * math.log(2 * math.pi))
        llhs -= .5 * torch.log(variances).sum(dim=-1)

        for flow_step in self.nnet_flow:
            new_means, new_variances = flow_step(flow, flow_params)
            flow = new_means + new_variances.sqrt() * flow
            llhs += -.5 * new_variances.log().sum(dim=-1)
        return flow, llhs

    def log_likelihood(self, data, params):
        '''The Normalizing flow layers cannot be used as an decoding
        distribution.
        '''
        raise NotImplementedError


__all__ = [
    'BernoulliLayer',
    'NormalDiagonalCovarianceLayer',
    'NormalIsotropicCovarianceLayer',
    'NormalIdentityCovarianceLayer',
    'InverseAutoRegressiveFlow',
    'BernoulliLayer',
]
