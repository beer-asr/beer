Models
======


Base Model
----------

``beer`` provides a collection of generative models that can be used in
various condition.

.. autoclass:: beer.BayesianModel
   :members:

.. autoclass:: beer.BayesianModelSet
   :members:


.. autoclass:: beer.BayesianParameter
   :members:

.. autoclass:: beer.BayesianParameterSet
   :members:


Concrete Models
---------------

+-----------------------------------------------+
| :any:`beer.NormalDiagonalCovariance`          |
+-----------------------------------------------+
| :any:`beer.NormalFullCovariance`              |
+-----------------------------------------------+
| :any:`beer.NormalDiagonalCovarianceSet`       |
+-----------------------------------------------+
| :any:`beer.NormalFullCovarianceSet`           |
+-----------------------------------------------+
| :any:`beer.NormalSetSharedDiagonalCovariance` |
+-----------------------------------------------+
| :any:`beer.NormalSetSharedDiagonalCovariance` |
+-----------------------------------------------+
| :any:`beer.Mixture`                           |
+-----------------------------------------------+
| :any:`beer.HMM`                               |
+-----------------------------------------------+

.. autoclass:: beer.NormalDiagonalCovariance
   :show-inheritance:
   :members: create

.. autoclass:: beer.NormalFullCovariance
   :show-inheritance:
   :members: create

.. autoclass:: beer.NormalDiagonalCovarianceSet
   :show-inheritance:
   :members: create

.. autoclass:: beer.NormalFullCovarianceSet
   :show-inheritance:
   :members: create

.. autoclass:: beer.NormalSetSharedDiagonalCovariance
   :show-inheritance:
   :members: create

.. autoclass:: beer.NormalSetSharedFullCovariance
   :show-inheritance:
   :members: create

.. autoclass:: beer.Mixture
   :show-inheritance:
   :members: create

.. autoclass:: beer.HMM
   :show-inheritance:
   :members: create