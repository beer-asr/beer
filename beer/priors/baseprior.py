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
        self.cache = {}

    def __repr__(self):
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            nparams=self.natural_hparams
        )

    def float(self):
        self.natural_parameters = self.natural_parameters.float()
        return self

    def double(self):
        self.natural_parameters = self.natural_parameters.double()
        return self

    def to(self, device):
        self.natural_parameters = self.natural_parameters.to(device)
        return self

    @property
    def natural_parameters(self):
        '``torch.Tensor``: Natural parameters.'
        return self._natural_params

    @natural_parameters.setter
    def natural_parameters(self, value):
        self._natural_params = value.detach()
        self.cache = {}

    def _to_std_parameters(self, natural_parameters=None):
        pass

    def to_std_parameters(self, natural_parameters=None):
        'Convert the natural parameters to their standard form.'
        if natural_parameters is not None:
            return self._to_std_parameters(natural_parameters)

        try:
            std_params = self.cache['std_params']
        except KeyError:
            std_params = self._to_std_parameters(natural_parameters)
            self.cache['std_params'] = std_params
        return std_params

    @abc.abstractmethod
    def to_natural_parameters(self, std_parameters=None):
        'Convert the standard parameters to their natural form.'
        pass


    @abc.abstractmethod
    def _expected_sufficient_statistics(self):
        pass

    def expected_sufficient_statistics(self):
        '''Expected value of the sufficient statistics of the
        distribution. This corresponds to the gradient of the
        log-normalizer w.r.t. the natural_parameters.

        Returns:
            ``torch.Tensor``
        '''
        try:
            exp_stats = self.cache['exp_stats']
        except KeyError:
            exp_stats = self._expected_sufficient_statistics()
            self.cache['exp_stats'] = exp_stats
        return exp_stats

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
    def _log_norm(self, natural_parameters=None):
        pass

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
        if natural_parameters is not None:
            return self._log_norm(natural_parameters)

        try:
            lnorm = self.cache['lnorm']
        except KeyError:
            lnorm = self._log_norm()
            self.cache['lnorm'] = lnorm
        return lnorm


__all__ = ['ExpFamilyPrior']
