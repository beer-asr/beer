Priors
======

Most of the models in ``beer`` uses the conjugacy property of the
exponential family of distribution.

Base Prior
----------

.. autoclass:: beer.ExpFamilyPrior
   :members:
   :special-members: __init__

Concrete Priors
---------------

.. autoclass:: beer.DirichletPrior
   :members:
   :special-members: __init__

.. autoclass:: beer.NormalGammaPrior
   :members:
   :special-members: __init__

.. autoclass:: beer.JointNormalGammaPrior
   :members:
   :special-members: __init__

.. autoclass:: beer.NormalWishartPrior
   :members:
   :special-members: __init__

.. autoclass:: beer.JointNormalWishartPrior
   :members:
   :special-members: __init__
