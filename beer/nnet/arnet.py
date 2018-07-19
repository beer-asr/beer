'''Module for Auto-Regressive Neural Network using mask.'''

import random
import torch
from .problayers import NormalDiagonalCovarianceLayer
from .neuralnetwork import parse_nnet_element
from .neuralnetwork import MergeTransform


def create_mask(ordering, max_connections):
    '''Create a mask for an (deep) Auto-Regressive model.

    Args:
        ordering (seq of int): Order for each dimension of the input.
        max_connections (seq of int): Maximum connection from the input
            to each hidden unit.

    Returns:
        ``torch.Tensor[len(max_connections):len(ordering)]``

    '''
    in_dim, out_dim = len(ordering), len(max_connections)
    retval = torch.zeros(out_dim, in_dim)
    for i, m_i in enumerate(max_connections):
        for j, order in enumerate(ordering):
            if m_i + 1 >= order + 1:
                retval[i, j] = 1
    return retval


def create_final_mask(ordering, max_connections):
    '''Create the final mask for an (deep) Auto-Regressive model.

    Args:
        ordering (seq of int): Order for each dimension of the output.
        max_connections (seq of int): Maximum connections of the last
            hidden layer.
        dtype (``torch.Tensor.dtype``): Type of the returned tensor.
        device (``torch.Tensor.device``): Device of the returned tensor.

    Returns:
        ``torch.Tensor[len(max_connections):len(ordering)]``

    '''
    out_dim, in_dim = len(ordering), len(max_connections)
    retval = torch.zeros(out_dim, in_dim)
    for i, order in enumerate(ordering):
        for j, m_j in enumerate(max_connections):
            if order + 1 > m_j + 1:
                retval[i, j] = 1
    return retval


class MaskedLinear(torch.nn.Module):
    '''Masked Linear transformation so that the dth output dimension
    depends only on a subset of the input.'''

    def __init__(self, mask, linear_transform):
        '''
        Args:
            mask (``torch.tensor``): Connection mask.
            linear_transform (``torch.nn.LinearTransform``): Linear
                transformation to mask.
        '''
        super().__init__()
        self.register_buffer('_mask', mask)
        self._linear_transform = linear_transform

    def forward(self, data):
        return torch.nn.functional.linear(
            data,
            self._linear_transform.weight * self._mask,
            self._linear_transform.bias
        )


class ARNetNormalDiagonalCovarianceLayer(torch.nn.Module):
    '''Output the mean and the diagonal covariance of a Normal
    density for a AR network.

    '''

    def __init__(self, mask, dim_in, dim_out):
        super().__init__()
        h2mean = torch.nn.Linear(dim_in, dim_out)
        self.h2mean = MaskedLinear(mask, h2mean)
        h2logvar = torch.nn.Linear(dim_in, dim_out)
        self.h2logvar =  MaskedLinear(mask, h2logvar)

    def forward(self, data):
        mean = self.h2mean(data)
        logvar = self.h2logvar(data)
        variance = 1e-2 + torch.nn.functional.sigmoid(logvar)
        return (1 - variance) * mean, variance


class SequentialMultipleInput(torch.nn.Sequential):
    def forward(self, *inputs):
        new_input = self[0](*inputs)
        if len(self) > 1:
            for module in self[1:]:
                new_input = module(new_input)
        return new_input


def create_arnetwork(conf):
    dim_in = conf['data_dim']
    context_dim = conf['context_dim']
    depth = conf['depth']
    width = conf['width']
    function_name, kwargs = parse_nnet_element(conf['activation'])
    function = getattr(torch.nn, function_name)
    activation = function(**kwargs)
    ordering = range(dim_in)
    layer_connections = []
    for i in range(depth):
        layer_connections.append(random.choices(range(0, dim_in - 1), k=width))
    arch = []
    previous_ordering = ordering
    previous_dim = dim_in
    for i in range(depth):
        ltrans = torch.nn.Linear(previous_dim, width)
        mask = create_mask(previous_ordering, layer_connections[i])
        if i > 0 or context_dim == 0:
            arch.append(MaskedLinear(mask, ltrans))
        else:
            arch.append(MergeTransform(MaskedLinear(mask, ltrans),
                                       torch.nn.Linear(context_dim, width)))
        arch.append(activation)
        previous_dim = width
        previous_ordering = layer_connections[i]

    # Final Normal layer.
    mask = create_final_mask(ordering, layer_connections[-1])
    arch.append(ARNetNormalDiagonalCovarianceLayer(mask, width, dim_in))

    return SequentialMultipleInput(*arch)
