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


class Identity(torch.nn.Module):
    def forward(self, data):
        return data


class NeuralNetworkBlock(torch.nn.Module):
    def __init__(self, structure, residual_connection=None):
        super().__init__()
        self.structure = structure
        self.residual_connection = residual_connection


    def forward(self, data):
        h = self.structure(data)
        if self.residual_connection is not None:
            h += self.residual_connection(data)
        return h


def load_value(strval, variables=None):
    '''Evaluate the string representation of a python type.

    Args:
        strval (string): String to interpret.
        variables (dictionary): Set of variables that will be replaced
            by their associated value before to interpret the python
            strings.
    '''
    if variables is None:
        variables = {}
    for variable, value in variables.items():
        strval = strval.replace(variable, str(value))
    return ast.literal_eval(strval)


def parse_nnet_element(strval):
    '''Parse a string definining a neural network element (layer,
    activation function, ...).

    The given string should be formatted as
    "object:param1=val1,param2=val2". For instance:

      Linear:in_features=10,out_features=20

    Args:
        strval (string): String defining the neural network element.

    Returns:
        ``torch.nn.<function>``: name of the object/function
        dict: Keyword arguments and their values (as string)

    '''
    str_kwargs = {}
    if ':' in strval:
        function_name, args_and_vals = strval.strip().split(':')
        for arg_and_val in args_and_vals.split(','):
            argname, str_argval = arg_and_val.split('=')
            str_kwargs[argname] = str_argval
    else:
        function_name = strval
    return function_name, str_kwargs


def create_nnet_element(strval, variables=None):
    '''Create a pytorch nnet element from a string.

    Args:
        strval (string): String defining the nnet element.
        variables (dictionary): Set of variables that will be replaced
            by their associated value before to interpret the python
            strings.

    Returs:
        ``torch.nn.<object>``

    '''
    function_name, str_kwargs = parse_nnet_element(strval)
    kwargs = {
        argname: load_value(arg_strval, variables)
        for argname, arg_strval in str_kwargs.items()
    }
    function = getattr(torch.nn, function_name)
    return function(**kwargs)


def create_nnet_block(block_conf, variables=None):
    '''Create a part of neural network.

    Args:
        block_conf (dict): Configuration dictionary.
        variables (dictionary): Set of variables that will be replaced
            by their associated value before to interpret the python
            strings.

    Returns:
        :any:`NeuralNetworkBlock`

    '''
    structure_list = [create_nnet_element(strval, variables)
                      for strval in block_conf['structure']]
    structure = torch.nn.Sequential(*structure_list)
    res_connection = block_conf['residual_connection']
    if res_connection == 'none':
        res_connection = None
    elif res_connection == 'identity':
        res_connection = Identity()
    else:
        res_connection = create_nnet_element(res_connection, variables)
    return NeuralNetworkBlock(structure, res_connection)


def create_chained_blocks(block_confs, variables):
    '''Create a chain of nnet blocks

    Args:
        block_confs (list of dict): Configuration dictionaries.
        variables (dictionary): Set of variables that will be replaced
            by their associated value before to interpret the python
            strings.

    Returns:
        list of :any:`NeuralNetworkBlock`

    '''
    return torch.nn.Sequential(*[create_nnet_block(block_conf, variables)
                                 for block_conf in block_confs])


def create_encoder(encoder_conf, variables):
    blocks = create_chained_blocks(encoder_conf['blocks'], variables)
    dim_in_normal_layer = load_value(encoder_conf['dim_input_normal_layer'],
                                     variables=variables)
    dim_out_normal_layer = load_value(encoder_conf['dim_output_normal_layer'],
                                      variables=variables)
    cov_type = encoder_conf['covariance']
    if cov_type == 'isotropic':
        normal_layer = NormalIsotropicCovarianceLayer(dim_in_normal_layer,
                                                      dim_out_normal_layer)
    elif cov_type == 'diagonal':
        normal_layer = NormalDiagonalCovarianceLayer(dim_in_normal_layer,
                                                        dim_out_normal_layer)
    else:
        raise ValueError('Unsupported covariance: {}'.format(cov_type))
    return torch.nn.Sequential(*blocks, normal_layer)


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
