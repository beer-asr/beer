'''Utility functions.'''

import torch


def onehot(labels, max_label):
    '''Convert a sequence of indices into a one-hot encoded matrix.

    Args:
        labels (seq): Sequence of indices (int) to convert.
        max_label (int): Maximum value for the index. This parameter
            defined the dimension of the returned matrix.

    Returns:
        ``torch.Tensor``: a matrix of N x `max_label` where each column \
            has a single element set to 1.
    '''
    retval = torch.zeros(len(labels), max_label)
    idxs = torch.range(0, len(labels) - 1).long()
    retval[idxs, labels] = 1
    return retval


def logsumexp(tensor, dim=0):
    '''Stable log -> sum -> exponential computation

    Args:
        tensor (``torch.Tensor``): Values.
        dim (int): Dimension along which to do the summation.

    Returns:
        ``torch.Tensor``
    '''
    tmax, _ = torch.max(tensor, dim=dim, keepdim=True)
    retval = tmax + (tensor - tmax).exp().sum(dim=dim, keepdim=True).log()
    new_size = list(tensor.size())
    del new_size[dim]
    return retval.view(*new_size)


__all__ = ['onehot', 'logsumexp']
