Variational AutoEncoder
=======================

    >>> import beer
    >>> import torch
    >>> from torch import nn
    >>> from torch.autograd import Variable
    >>> _ = torch.manual_seed(1)

Normal neural network with diagonal covariance
----------------------------------------------

Creation
^^^^^^^^

    >>> structure = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    >>> normal_mlp = beer.models.MLPNormalDiag(structure, hidden_dim=4,
    ...                                        target_dim=2)

Forwarding data through the neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> data = Variable(torch.randn(10, 2).type(torch.FloatTensor),
    ...                 requires_grad=False)
    >>> state = normal_mlp(data)
    >>> state['means'].size()
    torch.Size([10, 2])
    >>> state['means']
    Variable containing:
    -0.2483  0.1771
    -0.2310  0.0860
    -0.2668  0.3179
    -0.2623  0.2965
    -0.1969 -0.0088
    -0.1985  0.1049
    -0.2331  0.1677
    -0.2147  0.0192
    -0.1825  0.0060
    -0.1847  0.0269
    [torch.FloatTensor of size 10x2]
    <BLANKLINE>
    >>> state['logvars'].size()
    torch.Size([10, 2])
    >>> state['logvars']
    Variable containing:
     0.1039  0.2974
     0.0431  0.2078
     0.2508  0.4409
     0.2262  0.4295
     0.0577  0.1978
     0.1693  0.3195
     0.1343  0.3203
     0.0291  0.1725
     0.1093  0.2545
     0.1233  0.2704
    [torch.FloatTensor of size 10x2]
    <BLANKLINE>

Computing the log-likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> data = Variable(torch.randn(10, 2).type(torch.FloatTensor),
    ...                 requires_grad=False)
    >>> llh = normal_mlp.log_likelihood(data, state)
    >>> llh.size()
    torch.Size([10])
    >>> llh
    Variable containing:
    -5.6743
    -4.2903
    -4.0954
    -5.2909
    -4.8421
    -4.5772
    -4.6310
    -3.8785
    -4.0982
    -3.9613
    [torch.FloatTensor of size 10]
    <BLANKLINE>


Normal neural network with isotropic covariance matrix
------------------------------------------------------

Creation
^^^^^^^^

    >>> structure = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    >>> normal_mlp = beer.models.MLPNormalIso(structure, hidden_dim=4,
    ...                                       target_dim=2)

Forwarding data through the neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> data = Variable(torch.randn(10, 2).type(torch.FloatTensor),
    ...                 requires_grad=False)
    >>> state = normal_mlp(data)
    >>> state['means'].size()
    torch.Size([10, 2])
    >>> state['means']
    Variable containing:
     0.3028 -0.9357
     0.1137 -1.0394
     0.2545 -0.9833
     0.2541 -0.9864
     0.2309 -0.9403
     0.3209 -0.9272
     0.4174 -0.8551
     0.3474 -0.9138
     0.1928 -1.0248
     0.1774 -0.9597
    [torch.FloatTensor of size 10x2]
    <BLANKLINE>
    >>> state['logvars'].size()
    torch.Size([10, 1])
    >>> state['logvars']
    Variable containing:
    -0.5776
    -0.5345
    -0.5450
    -0.5418
    -0.5972
    -0.5788
    -0.5993
    -0.5800
    -0.5224
    -0.5922
    [torch.FloatTensor of size 10x1]
    <BLANKLINE>

Computing the log-likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> data = Variable(torch.randn(10, 2).type(torch.FloatTensor),
    ...                 requires_grad=False)
    >>> llh = normal_mlp.log_likelihood(data, state)
    >>> llh.size()
    torch.Size([10])
    >>> llh
    Variable containing:
    -4.3139
    -4.2653
    -4.0694
    -4.9310
    -3.4181
    -2.6376
    -4.8554
    -5.6976
    -4.2100
    -4.9598
    [torch.FloatTensor of size 10]
    <BLANKLINE>


Normal neural network with full covariance matrix
-------------------------------------------------

Creation
^^^^^^^^

    >>> structure = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    >>> normal_mlp = beer.models.MLPNormalFull(structure, hidden_dim=4,
    ...                                        target_dim=2)

Forwarding data through the neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> data = Variable(torch.randn(10, 2).type(torch.FloatTensor),
    ...                 requires_grad=False)
    >>> state = normal_mlp(data)
    >>> state['means'].size()
    torch.Size([10, 2])
    >>> state['means']
    Variable containing:
    -0.2096  0.6874
    -0.2298  0.6833
    -0.2631  0.7251
    -0.2288  0.7055
    -0.2528  0.6503
    -0.2194  0.6556
    -0.2590  0.6488
    -0.2831  0.6747
    -0.1928  0.7625
    -0.2311  0.6686
    [torch.FloatTensor of size 10x2]
    <BLANKLINE>
    >>> state['logvars'].size()
    torch.Size([10, 2])
    >>> state['logvars']
    Variable containing:
     0.1288  0.3899
     0.1527  0.3370
     0.2673  0.1664
     0.1910  0.2928
     0.1285  0.3382
     0.0796  0.4468
     0.1362  0.3206
     0.2153  0.1988
     0.2368  0.2822
     0.1272  0.3669
    [torch.FloatTensor of size 10x2]
    <BLANKLINE>
    >>> state['L_tris'].size()
    torch.Size([10, 1])
    >>> state['L_tris']
    Variable containing:
    -0.1150
    -0.0670
     0.1309
    -0.0066
    -0.0941
    -0.1873
    -0.0789
     0.0585
     0.0482
    -0.1068
    [torch.FloatTensor of size 10x1]
    <BLANKLINE>

Computing the log-likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> data = Variable(torch.randn(10, 2).type(torch.FloatTensor),
    ...                 requires_grad=False)
    >>> llh = normal_mlp.log_likelihood(data, state)
    >>> llh.size()
    torch.Size([10])
    >>> llh
    Variable containing:
    -6.0740
    -3.8108
    -2.9631
    -3.2364
    -7.1188
    -3.0280
    -7.3108
    -4.6933
    -3.9187
    -7.9141
    [torch.FloatTensor of size 10]
    <BLANKLINE>

