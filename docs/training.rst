Training
========

In ``beer``, all the models can be trained withing the Variational Bayes
Inference (VBI). In this framework, the objective function to optimize
is the Evidence Lower BOund (ELBO) and it is defined as:

.. math::
   \mathcal{L}(\theta) = \langle \ln p(x | \theta)
        \rangle_{q(\theta)} - \text{D}(q(\theta) || p(\theta))


Evidence Lower Bound Objective function
---------------------------------------

.. autoclass:: beer.EvidenceLowerBound

Optimizer
---------

.. autoclass:: beer.BayesianModelOptimizer
   :members: