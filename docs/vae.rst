Variational AutoEncoder
=======================

    >>> import beer
    >>> import torch
    >>> from torch import nn
    >>> from torch.autograd import Variable

Generate some random data for the sake of the example:

    >>> data = Variable(torch.randn(10, 2).type(torch.FloatTensor),
    ...                 requires_grad=False)

Normal neural network with diagonal covariance
----------------------------------------------

Creation
^^^^^^^^

    >>> _ = torch.manual_seed(1)
    >>> structure = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    >>> normal_mlp = beer.models.MLPNormalDiag(structure, hidden_dim=4,
    ...                                        target_dim=2)

Forwarding data through the neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> state = normal_mlp(data)
    >>> state['means'].size()
    torch.Size([10, 2])
    >>> state['logvars'].size()
    torch.Size([10, 2])

Computing the log-likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> llh = normal_mlp.log_likelihood(data, state)
    >>> llh.size()
    torch.Size([10])

Sampling
^^^^^^^^

    >>> _ = torch.manual_seed(1)
    >>> samples= normal_mlp.sample(state)
    >>> samples.size()
    torch.Size([10, 2])

Normal neural network with isotropic covariance matrix
------------------------------------------------------

Creation
^^^^^^^^

    >>> _ = torch.manual_seed(1)
    >>> structure = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    >>> normal_mlp = beer.models.MLPNormalIso(structure, hidden_dim=4,
    ...                                       target_dim=2)

Forwarding data through the neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> state = normal_mlp(data)
    >>> state['means'].size()
    torch.Size([10, 2])
    >>> state['logvars'].size()
    torch.Size([10, 1])

Computing the log-likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> llh = normal_mlp.log_likelihood(data, state)
    >>> llh.size()
    torch.Size([10])

Sampling
^^^^^^^^

    >>> _ = torch.manual_seed(1)
    >>> samples= normal_mlp.sample(state)
    >>> samples.size()
    torch.Size([10, 2])


Normal neural network with full covariance matrix
-------------------------------------------------

Creation
^^^^^^^^

    >>> structure = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    >>> normal_mlp = beer.models.MLPNormalFull(structure, hidden_dim=4,
    ...                                        target_dim=2)

Forwarding data through the neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> state = normal_mlp(data)
    >>> state['means'].size()
    torch.Size([10, 2])
    >>> state['logprecs'].size()
    torch.Size([10, 2])
    >>> state['L_tris'].size()
    torch.Size([10, 1])

Computing the log-likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> llh = normal_mlp.log_likelihood(data, state)
    >>> llh.size()
    torch.Size([10])

Sampling
^^^^^^^^

    >>> _ = torch.manual_seed(1)
    >>> samples= normal_mlp.sample(state)
    >>> samples.size()
    torch.Size([10, 2])
