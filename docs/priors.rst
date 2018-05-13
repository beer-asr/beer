Conjugate Priors
================

Most of the models in ``beer`` are *Bayesian* in the sense that they
have a distribution over their parameter, usually called the *prior*.
Also, these models are trained with the Variable Bayes Inference
framework leading to a *posterior* distribution over the
aforementioned parameters. The conjugate priors are special distribution
that have the particularity (given a certain type of model) to have the
same parametric from as the posterior counterpart. The prior
distribution object described here can thus be used either as prior
and posterior.

Base Prior
----------

.. autoclass:: beer.ExpFamilyPrior
   :members:
   :special-members: __init__

Concrete Priors
---------------

+---------------------------------------------+
| Implemented priors                          |
+=============================================+
| :any:`beer.DirichletPrior`                  |
+---------------------------------------------+
| :any:`beer.NormalGammaPrior`                |
+---------------------------------------------+
| :any:`beer.JointNormalGammaPrior`           |
+---------------------------------------------+
| :any:`beer.NormalWishartPrior`              |
+---------------------------------------------+
| :any:`beer.JointNormalWishartPrior`         |
+---------------------------------------------+
| :any:`beer.NormalFullCovariancePrior`       |
+---------------------------------------------+
| :any:`beer.NormalIsotropicCovariancePrior`  |
+---------------------------------------------+
| :any:`beer.MatrixNormalPrior`               |
+---------------------------------------------+
| :any:`beer.GammaPrior`                      |
+---------------------------------------------+

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

.. autoclass:: beer.NormalFullCovariancePrior
   :members:
   :special-members: __init__

.. autoclass:: beer.NormalIsotropicCovariancePrior
   :members:
   :special-members: __init__

.. autoclass:: beer.MatrixNormalPrior
   :members:
   :special-members: __init__

.. autoclass:: beer.GammaPrior
   :members:
   :special-members: __init__