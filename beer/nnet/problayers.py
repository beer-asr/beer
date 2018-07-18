'''Neural Network output layer that gives the parameters of a
probability distribution.'''

import torch


class NormalDiagonalCovarianceLayer(torch.nn.Module):
    '''Output the mean and the diagonal covariance of a Normal
    density.

    '''

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)
        self.h2logvar = torch.nn.Linear(dim_in, dim_out)

    def forward(self, data):
        mean = self.h2mean(data)
        logvar = self.h2logvar(data)
        variance = 1e-2 * torch.nn.functional.softplus(logvar)
        return mean, variance


class NormalIsotropicCovarianceLayer(torch.nn.Module):
    '''Output the mean and the covariance of a Normal density with
    isotropic covariance matrix.

    '''

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)
        self.h2logvar = torch.nn.Linear(dim_in, 1)
        self.out_dim = dim_out

    def forward(self, data):
        mean = self.h2mean(data)
        logvar = self.h2logvar(data)
        variance = 1e-2 * torch.nn.functional.softplus(logvar)
        return mean, variance * torch.ones(1, self.out_dim, dtype=data.dtype,
                                           device=data.device)


class NormalUnityCovarianceLayer(torch.nn.Module):
    '''Output the mean a Normal density with unity covariance
    matrix.

    '''

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)

    def forward(self, data):
        mean = self.h2mean(data)
        return [mean]


class BernoulliLayer(torch.nn.Module):
    '''Output the mean of a Bernoulli distribution.'''

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        mean = self.h2mean(data)
        return [self.sigmoid(mean)]


class BetaLayer(torch.nn.Module):
    '''Output the two shape parameters (alphas and betas) for a set of
    Beta distribution.'''


    def __init__(self, dim_in, dim_out, min_value=1e-1, max_value=10):
        super().__init__()
        self.h2alpha = torch.nn.Linear(dim_in, dim_out)
        self.h2beta = torch.nn.Linear(dim_in, dim_out)
        self.sigmoid = torch.nn.Sigmoid()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, data):
        alpha = self.min_value + self.max_value * self.sigmoid(self.h2alpha(data))
        beta = self.min_value + self.max_value * self.sigmoid(self.h2beta(data))
        return alpha, beta


class NormalizingFlowLayer(torch.nn.Module):
    '''Output the parameters of the initial normal density and the
    parameters of the flow.'''


    def __init__(self, dim_in, flow_params_dim, normal_layer):
        super().__init__()
        self.normal_layer = normal_layer
        self.h2flow = torch.nn.Linear(dim_in, flow_params_dim)

    def forward(self, data):
        flow_params = self.h2flow(data)
        normal_params = self.normal_layer(data)
        return (*normal_params, flow_params)


def create(layer_conf):
    layer_type = layer_conf['type']
    in_dim = layer_conf['dim_in']
    out_dim = layer_conf['dim_out']

    if layer_type == 'NormalizingFlowLayer':
        cov_type = layer_conf['covariance']
        flow_params_dim = layer_conf['flow_params_dim']
        if cov_type == 'isotropic':
            init_normal_layer = NormalIsotropicCovarianceLayer(in_dim, out_dim)
        elif cov_type == 'diagonal':
            init_normal_layer = NormalDiagonalCovarianceLayer(in_dim, out_dim)
        else:
            raise ValueError('Unsupported covariance type: {}'.format(cov_type))
        return NormalizingFlowLayer(in_dim, flow_params_dim, init_normal_layer)
    elif layer_type == 'NormalLayer':
        cov_type = layer_conf['covariance']
        if cov_type == 'isotropic':
            return NormalIsotropicCovarianceLayer(in_dim, out_dim)
        elif cov_type == 'diagonal':
            return NormalDiagonalCovarianceLayer(in_dim, out_dim)
        elif cov_type == 'unity':
            return NormalUnityCovarianceLayer(in_dim, out_dim)
        else:
            raise ValueError('Unsupported covariance type: {}'.format(cov_type))
    elif layer_type == 'BetaLayer':
        return BetaLayer(in_dim, out_dim)
    elif layer_type == 'BernoulliLayer':
        return BernoulliLayer(in_dim, out_dim)
    else:
        raise ValueError('Unknown probability layer type: {}'.format(layer_type))


__all__ = [
    'NormalDiagonalCovarianceLayer',
    'NormalIsotropicCovarianceLayer',
    'NormalUnityCovarianceLayer',
    'NormalizingFlowLayer',
    'BernoulliLayer',
    'BetaLayer'
]
