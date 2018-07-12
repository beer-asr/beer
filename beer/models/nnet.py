'''Specific Neural Network architectures.'''

import ast
import torch
from .bayesmodel import BayesianModel


class NeuralNetwork(torch.nn.Module):
    '''Generic Neural network class.'''

    def __init__(self, seqs):
        super().__init__()
        self.seq = torch.nn.Sequential(*seqs)

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


class BernoulliLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2mean = torch.nn.Linear(dim_in, dim_out)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        mean = self.h2mean(data)
        return self.sigmoid(mean)


class BetaLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.h2alpha = torch.nn.Linear(dim_in, dim_out)
        self.h2beta = torch.nn.Linear(dim_in, dim_out)
        self.sigmoid = torch.nn.Sigmoid

    def forward(self, data):
        alpha = self.h2alpha(data).exp()
        beta = self.h2beta(data).exp()
        return alpha, beta


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


def load_value(value, variables=None):
    '''Evaluate the string representation of a python type if the given
    value is a string.

    Args:
        value (object): value to load w/o interpretation.
        variables (dictionary): Set of variables that will be replaced
            by their associated value before to interpret the python
            strings.
    '''
    if not isinstance(value, str):
        return value

    if variables is None:
        variables = {}
    for variable, val in variables.items():
        value = value.replace(variable, str(val))
    return ast.literal_eval(value)


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


def create_encoder(encoder_conf, dtype, device, variables):
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
    retval = torch.nn.Sequential(*blocks, normal_layer)
    return retval.type(dtype).to(device)


def create_normal_decoder(decoder_conf, dtype, device, variables):
    blocks = create_chained_blocks(decoder_conf['blocks'], variables)
    dim_in_model_layer = load_value(decoder_conf['dim_input_model_layer'],
                                     variables=variables)
    dim_out_model_layer = load_value(decoder_conf['dim_output_model_layer'],
                                      variables=variables)
    normal_layer = NormalUnityCovarianceLayer(dim_in_model_layer,
                                              dim_out_model_layer)
    retval = torch.nn.Sequential(*blocks, normal_layer)
    return retval.type(dtype).to(device)


def create_bernoulli_decoder(decoder_conf, dtype, device, variables):
    blocks = create_chained_blocks(decoder_conf['blocks'], variables)
    dim_in_model_layer = load_value(decoder_conf['dim_input_model_layer'],
                                     variables=variables)
    dim_out_model_layer = load_value(decoder_conf['dim_output_model_layer'],
                                      variables=variables)
    bernoulli_layer = BernoulliLayer(dim_in_model_layer, dim_out_model_layer)
    retval = torch.nn.Sequential(*blocks, bernoulli_layer)
    return retval.type(dtype).to(device)


def create_beta_decoder(decoder_conf, dtype, device, variables):
    blocks = create_chained_blocks(decoder_conf['blocks'], variables)
    dim_in_model_layer = load_value(decoder_conf['dim_input_model_layer'],
                                     variables=variables)
    dim_out_model_layer = load_value(decoder_conf['dim_output_model_layer'],
                                      variables=variables)
    beta_layer = BetaLayer(dim_in_model_layer, dim_out_model_layer)
    retval = torch.nn.Sequential(*blocks, beta_layer)
    return retval.type(dtype).to(device)


# This module does not have a public interface.
__all__ = []
