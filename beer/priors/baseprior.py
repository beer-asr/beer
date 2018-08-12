'''Base implementation of a prior from the exponential family of
distribution.'''

import abc
import torch
import torch.autograd as ta


def _bregman_divergence(f_val1, f_val2, grad_f_val2, val1, val2):
    return f_val1 - f_val2 - torch.sum(grad_f_val2 * (val1 - val2))


class ExpFamilyPrior(metaclass=abc.ABCMeta):
    '''Abstract base class for (conjugate) priors from the exponential
    family of distribution.

    '''
    __repr_str = '{classname}(natural_params={nparams})'

    @staticmethod
    def kl_div(model1, model2):
        '''Kullback-Leibler divergence between two densities of the same
        type from the exponential family of distribution.
        Args:
            model1 (:any:`beer.ExpFamilyPrior`): First model.
            model2 (:any:`beer.ExpFamilyPrior`): Second model.

        Returns
            float: Value of te KL. divergence between these two models.

        '''
        return _bregman_divergence(
            model2.log_norm().detach(),
            model1.log_norm().detach(),
            model1.expected_sufficient_statistics(),
            model2.natural_parameters,
            model1.natural_parameters
        )

    def __init__(self, natural_parameters):
        '''Initialize the base class.

        Args:
            natural_parameters (``torch.Tensor``): Natural parameters of
                the distribution.
        '''
        self._natural_params = natural_parameters.detach()

    def __repr__(self):
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            nparams=self.natural_hparams
        )

    @property
    def natural_parameters(self):
        '``torch.Tensor``: Natural parameters.'
        return self._natural_params

    @natural_parameters.setter
    def natural_parameters(self, value):
        self._natural_params = value.detach()

    @property
    @abc.abstractmethod
    def strength(self):
        'Strength of the distribution in term of observed counts.'
        pass

    @strength.setter
    @abc.abstractmethod
    def strength(self, value):
        pass

    @abc.abstractmethod
    def to_std_parameters(self, natural_parameters=None):
        'Convert the natural parameters to their standard form.'
        pass

    @abc.abstractmethod
    def to_natural_parameters(self, std_parameters=None):
        'Convert the standard parameters to their natural form.'
        pass


    @abc.abstractmethod
    def expected_sufficient_statistics(self):
        '''Expected value of the sufficient statistics of the
        distribution. This corresponds to the gradient of the
        log-normalizer w.r.t. the natural_parameters.

        Returns:
            ``torch.Tensor``
        '''
        pass

    def expected_value(self):
        '''Mean value of the random variable w.r.t. to the distribution.

        Returns:
            ``torch.Tensor``
        '''
        copied_tensor = torch.tensor(self.natural_parameters,
                                     requires_grad=True)
        log_norm = self.log_norm(copied_tensor)
        ta.backward(log_norm)
        return copied_tensor.grad.detach()

    @abc.abstractmethod
    def log_norm(self, natural_parameters=None):
        '''Abstract method to be implemented by subclasses of
        ``beer.ExpFamilyPrior``.

        Log-normalizing function of the density given the current
        parameters.

        Args:
            natural_parameters (``torch.Tensor``): If provided,
             compute the log-normalizer of the distribution with
             the given natural parameters.

        Returns:
            ``torch.Tensor[1]`` : Log-normalization value.

        '''
        pass


__all__ = ['ExpFamilyPrior']
