'''Utility functions.'''

import subprocess
import io
import os
import torch
import torch.autograd as autograd


__all__ = ['reserve_gpu', 'onehot', 'logsumexp', 'symmetrize_matrix',
           'make_symposdef', 'sample_from_normals', 'jacobians',
           'approximate_hessian']


def get_gpus():
    'Return a dict uuid -> index for all the GPUs on the local matchine.'
    cmd = 'nvidia-smi --query-gpu=gpu_uuid,index --format=csv,noheader'
    outproc = subprocess.run(cmd, shell=True, text=True,
                             capture_output=True, check=True)
    output = io.StringIO(outproc.stdout)
    retval = {}
    for line in output:
        tokens = line.strip().split(',')
        uuid, idx = tokens[0], int(tokens[1])
        retval[uuid] = idx
    return retval

def get_active_gpus():
    'Return a set of uuids corresponding to all the GPUs being used.'
    cmd = 'nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader'
    outproc = subprocess.run(cmd, shell=True, text=True,
                             capture_output=True, check=True)
    output = io.StringIO(outproc.stdout)
    return set([line.strip() for line in output])

# Error raised when the "reserve_gpu" fails.
class NoGPUAvailable(Exception): pass

def reserve_gpu(max_retry=12, logger=None, wait_time=10):
    '''Reserve a GPU in adversarial environment.

    This function assumes CUDA is installed and the `nvidia-smi` tool
    is available in the PATH.

    Args:
        max_retry (int): Number of attemps before failing.
        logger (logger): Logger which will log (debug messages) the
            failed attempt to reserve the GPUs.

    Returns:
        (int): index of the GPU reserved.

    '''
    import time
    cmd = "nvidia-smi --query-gpu=index,utilization.gpu --format='csv'"
    count = 0
    gpu_reserved = False
    gpus = get_gpus()
    while count < max_retry and not gpu_reserved:
        active_gpus = get_active_gpus()
        for gpu, gpu_idx in gpus.items():
            if gpu not in active_gpus:
                logger.debug(f'trying to reserve gpu {gpu_idx}')
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
                # Reserve the GPU by allocating some dummy data.
                try:
                    dummy = torch.zeros(1, device=torch.device('cuda'))
                    gpu_reserved = True
                except Exception as err:
                    if logger is not None:
                        logger.debug(f'attempt to reserve a GPU failed: {err}')
            if gpu_reserved:
                break
        else:
            logger.debug(f'waiting {wait_time} seconds...')
            time.sleep(wait_time)
        count += 1

    if not gpu_reserved:
        raise NoGPUAvailable('Couldn\'t reserve a GPU')
    return  gpu_idx


def onehot(labels, max_label, dtype, device):
    '''Convert a sequence of indices into a one-hot encoded matrix.

    Args:
        labels (seq): Sequence of indices (int) to convert.
        max_label (int): Maximum value for the index. This parameter
            defined the dimension of the returned matrix.
        dtype (``torch.dtype``): Data type of the return tensor.
        device (``torch.devce``): On which device to allocate the
            tensor.

    Returns:
        ``torch.Tensor``: a matrix of N x `max_label` where each column \
            has a single element set to 1.
    '''
    retval = torch.zeros(len(labels), max_label, dtype=dtype, device=device)
    idxs = torch.arange(0, len(labels)).long()
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
    retval = torch.where(
        (tmax == float('-inf')) | (tmax == float('inf')),
         tmax,
         tmax + (tensor - tmax).exp_().sum(dim=dim, keepdim=True).log_()
         )
    new_size = list(tensor.size())
    del new_size[dim]
    return retval.view(*new_size)


def symmetrize_matrix(mat):
    '''Enforce a matrix to be symmetric.

    Args:
        mat (``torch.Tensor[dim, dim]``): Matrix to symmetrize

    Returns:
        ``torch.Tensor[dim, dim]``

    '''
    return .5 * (mat + mat.t())


def make_symposdef(mat, eval_threshold=1e-1):
    '''Enforce a matrix to be symmetric and positive definite.

    Args:
        mat (``torch.Tensor[dim, dim]``): Input matrix.
        eval_threshold (float): Minimum value of the eigen values of
            the matrix.

    Returns:
        ``torch.Tensor[dim, dim]``

    '''
    sym_mat = symmetrize_matrix(mat)
    evals, evecs = torch.symeig(sym_mat, eigenvectors=True)

    threshold = torch.tensor(eval_threshold, dtype=sym_mat.dtype,
                             device=sym_mat.device)
    new_evals = torch.where(
        evals < threshold * evals[-1],
        threshold * evals[-1],
        evals
    )
    return (evecs @ torch.diag(new_evals) @ evecs.t()).view(*mat.shape)


def sample_from_normals(means, variances, nsamples):
    '''Sample for a set of Normal distribution with diagonal covariance
    using the re-parameterization trick. The gradient of the sampled values
    (and their subsequent transformations) can be computed in the same
    way as any ``torch.Tensor``.

    Args:
        means (``torch.Tensor[N,D]``): A set of N means of D dimensions.
        variances (``torch.Tensor[N,D]``): Diagonal of the covariance matrix
            for each distribution.
        nsamples (int): Number of samples per distribution.

    Returns:
        (``torch.Tensor[nsamples,N,D]``): sampled values.

    '''
    dim1, dim2 = means.shape
    return means + torch.sqrt(variances) * torch.randn(nsamples, dim1, dim2,
                                                       dtype=means.dtype,
                                                       device=means.device)


def jacobians(outputs, inputs):
    '''Compute the jacobians for each input tensor.

    Args:
        ouputs (``torch.Tensor[N]``): Output tensor.
        inputs (sequence): Sequence of input tensors.

    Returns:
        sequence: A jacobian matrix for each input tensor.

    '''
    dim, dtype, device = len(outputs), outputs.dtype, outputs.device
    jac_matrices = [torch.zeros(dim, len(tensor.view(-1)), dtype=dtype,
                    device=device) for tensor in inputs]
    for input_tensor in inputs:
        input_tensor.grad = None
    for i in range(dim):
        autograd.backward(outputs[i], retain_graph=True)
        for j, in_tensor in enumerate(inputs):
            jac_matrices[j][i, :] = in_tensor.grad.view(-1)
    return jac_matrices


def approximate_hessian(list_grads):
    '''Outer product approximation of the Hessian.

    Args:
        grads (list of ``torch.Tensor[N, dim]``): list of sequence of
            gradients.

    Returns:
        list of ``torch.Tensor[dim, dim]``: Approximate Hessian matrices.

    '''
    hessians = []
    for grads in list_grads:
        hessians.append(torch.sum(grads[:, :, None] * grads[:, None, :], dim=0))
    return hessians

