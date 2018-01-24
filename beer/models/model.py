
'''Abstract Base Class for a model.'''

import abc
import numpy as np


class Model(metaclass=abc.ABCMeta):
    """Abstract Base Class for all beer's model."""

    @abc.abstractmethod
    def _fit_step(self, mini_batch):
        NotImplemented


class ConjugateExponentialModel(metaclass=abc.ABCMeta):
    '''Abstract base class for Conjugate Exponential models.'''

    pass
