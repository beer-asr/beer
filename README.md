BEER: the Bayesian Speech Recognizer
====================================

Beer is a toolkit that provide Bayesian machine learning tools for
speech technologies.

Beer is currently under construction and many things are subject to
change !

Requirements
------------

Beer is built upon the [pytorch](http://pytorch.org) and several other
third party packages. To make sure that all the dependencies are
installed we recommend to create a new anaconda environment with
the given environment file:

```
    $ conda env create -f condaenv.yml
```

This will create a new environment name `beer`.

Installation
------------

Assuming that you have already created the `beer` environment, type
in a terminal.

```
  $ source activate beer
  $ python setup.py install
```

This will install ``beer`` and eventually ``numpy`` and ``sicpy``
if there are not install yet. Note that it will not
automatically install ``pytorch``. Please refer to the
[pytorch documentation](https://github.com/pytorch/pytorch)
to install it.


Usage
-----

Have a look to our [examples](https://github.com/beer-asr/beer/tree/master/examples)
to get started with beer.
