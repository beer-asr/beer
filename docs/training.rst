Training
========

In ``beer``, all the models can be trained with the Variational Bayes
Inference (VBI). In this framework, the objective function to optimize
is the Evidence Lower BOund (ELBO) and it is defined as:

.. math::
   \mathcal{L}(\theta) = \langle \ln p(x | \theta)
        \rangle_{q(\theta)} - \text{D}(q(\theta) || p(\theta))


Bayesian parameters which have a conjugate prior are optimize through
a *natural gradient* ascent whereas other "standard" parameters (for
instance the neural network's parameters in a Variational AutoEncoder
are optimized by the standard gradient ascent).

Evidence Lower Bound Objective function
---------------------------------------

.. autoclass:: beer.EvidenceLowerBound

Optimizer
---------

.. autoclass:: beer.BayesianModelOptimizer
   :members: