'''Implementation of a generic Neural Network composed of blocks
(residual or not).

'''

import ast
import torch


class ReshapeLayer(torch.nn.Module):
    '''This layer is just a convenience so that the user can parameterize
    how the data should be presented to the neural network or, to
    alternate between simple feed-forward and convolutional layers.

    '''
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, data):
        return data.view(*self.shape)


class NeuralNetworkBlock(torch.nn.Module):
    def __init__(self, structure, residual_connection=False):
        super().__init__()
        self.structure = structure
        self.residual_connection = residual_connection

    def forward(self, data):
        h = self.structure(data)
        if self.residual_connection:
            h += data
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
        for arg_and_val in args_and_vals.split(';'):
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
    if hasattr(torch.nn, function_name):
        function = getattr(torch.nn, function_name)
    elif function_name == 'ReshapeLayer':
        function = ReshapeLayer
    else:
        raise ValueError('Unknown nnet element type: {}'.format(function_name))
    kwargs = {
        argname: load_value(arg_strval, variables)
        for argname, arg_strval in str_kwargs.items()
    }
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
                      for strval in block_conf['block_structure']]
    structure = torch.nn.Sequential(*structure_list)
    res_connection = block_conf['residual']
    return NeuralNetworkBlock(structure, res_connection)


def create(nnet_blocks_conf, dtype, device, variables):
    '''Create a neural network.

    Args:
        nnet_blocks_conf (list of dict): Configuration dictionaries.
        variables (dictionary): Set of variables that will be replaced
            by their associated value before to interpret the python
            strings from the configuration data.

    Returns:
        list of :any:`NeuralNetworkBlock`

    '''
    network = torch.nn.Sequential(*[create_nnet_block(block_conf, variables)
                                 for block_conf in nnet_blocks_conf])
    return network.type(dtype).to(device)


# This module does not have a public interface.
__all__ = []
