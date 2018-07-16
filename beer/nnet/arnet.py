'''Module for Auto-Regressive Neural Network.'''

import torch


def create_mask(ordering, max_connections, dtype, device):
    '''Create a mask for an (deep) Auto-Regressive model.

    Args:
        ordering (seq of int): Order for each dimension of the input.
        max_connections (seq of int): Maximum connection from the input
            to each hidden unit.
        dtype (``torch.Tensor.dtype``): Type of the returned tensor.
        device (``torch.Tensor.device``): Device of the returned tensor.

    Returns:
        ``torch.Tensor[len(max_connections):len(ordering)]``

    '''
    in_dim, out_dim = len(ordering), len(max_connections)
    retval = torch.zeros(out_dim, in_dim, dtype=dtype, device=device)
    for i, m_i in enumerate(max_connections):
        for j, order in enumerate(ordering):
            if m_i >= order:
                retval[i, j] = 1
    return retval


def create_final_mask(ordering, max_connections, dtype, device):
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
    retval = torch.zeros(out_dim, in_dim, dtype=dtype, device=device)
    for i, order in enumerate(ordering):
        for j, m_j in enumerate(max_connections):
            if order > m_j:
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
        self._mask = mask
        self._linear_transform = linear_transform

    def forward(self, data):
        return torch.nn.functional.linear(
            data,
            self._linear_transform.weight * self._mask,
            self._linear_transform.bias
        )

    def float(self):
        self._linear_transform = self._linear_transform.float()
        self._mask = self._mask.float()

    def double(self):
        self._linear_transform = self._linear_transform.double()
        self._mask = self._mask.double()

    def to(self, device):
        self._linear_transform = self._linear_transform.to(device)
        self._mask = self._mask.to(device)
