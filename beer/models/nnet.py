'''Specific Neural Network architectures.'''

import ast
import torch
from .bayesmodel import BayesianModel


non_linearity_fns = {
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid()
}


class NeuralNetwork(torch.nn.Module):
    '''Generic Neural network class.'''

    def __init__(self, seqs):
        super().__init__()
        self.seq = torch.nn.Sequential(*seqs)

    def forward(self, data):
        return self.seq(data)


class FeedForward(torch.nn.Module):
    '''Simple fully connected MLP.'''

    def __init__(self, dim_in, dim_out, dim_hlayer, n_layer, non_linearity):
        super().__init__()
        layers = []
        last_dim = dim_in
        for _ in range(n_layer - 1):
            layers.append(torch.nn.Linear(last_dim, dim_hlayer))
            layers.append(non_linearity_fns[non_linearity])
            last_dim = dim_hlayer
        layers.append(torch.nn.Linear(last_dim, dim_out))
        layers.append(non_linearity_fns[non_linearity])
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, data):
        return self.seq(data)


class NormalDiagonalCovarianceLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)
        self.h2logvar = torch.nn.Linear(dim_in, dim_out)

    def forward(self, data):
        mean = self.h2mean(data)
        logvar = self.h2logvar(data)
        return mean, logvar.exp()


class NormalIsotropicCovarianceLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)
        self.h2logvar = torch.nn.Linear(dim_in, 1)
        self.out_dim = dim_out

    def forward(self, data):
        mean = self.h2mean(data)
        logvar = self.h2logvar(data)
        return mean, logvar.exp() * torch.ones(1, self.out_dim,
                                               dtype=data.dtype,
                                               device=data.device)


class NormalUnityCovarianceLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)

    def forward(self, data):
        mean = self.h2mean(data)
        return mean


def load_value(strval, variables={}):
    '''Evaluate the string representation of a python type.

    Args:
        strval (string): String to interpret.
        variable (dictionary): Set of variables that will be replaced
            by their associated value before to interpret the python
            strings.
    '''
    for variable, value in variables.items():
        strval = strval.replace(variable, str(value))
    return ast.literal_eval(strval)


def _create_block(block_conf, tensor_type):
    block_type = block_conf['type']
    if block_type == 'FeedForwardEncoder':
        dim_in = block_conf['dim_in']
        dim_out = block_conf['dim_out']
        dim_hlayer = block_conf['dim_hlayer']
        cov_type = block_conf['covariance']
        n_layer = block_conf['n_layer']
        non_linearity = block_conf['non_linearity']
        nnet = FeedForward(dim_in, dim_hlayer, dim_hlayer, n_layer, non_linearity)
        if cov_type == 'isotropic':
            normal_layer = NormalIsotropicCovarianceLayer(dim_hlayer, dim_out)
        elif cov_type == 'diagonal':
            normal_layer = NormalDiagonalCovarianceLayer(dim_hlayer, dim_out)
        else:
            raise ValueError('Unsupported covariance: {}'.format(cov_type))
        retval = NeuralNetwork([nnet, normal_layer]).type(tensor_type)
        return retval
    elif block_type == 'FeedForwardDecoder':
        dim_in = block_conf['dim_in']
        dim_out = block_conf['dim_out']
        dim_hlayer = block_conf['dim_hlayer']
        n_layer = block_conf['n_layer']
        non_linearity = block_conf['non_linearity']
        nnet = FeedForward(dim_in, dim_hlayer, dim_hlayer, n_layer, non_linearity)
        normal_layer = NormalUnityCovarianceLayer(dim_hlayer, dim_out)
        retval = NeuralNetwork([nnet, normal_layer]).type(tensor_type)
        return retval
    else:
        raise ValueError('Unsupported architecture: {}'.format(block_type))


def create(model_conf, mean, variance, create_model_handle):
    return _create_block(model_conf, mean.dtype)

__all__ = ['FeedForward']
