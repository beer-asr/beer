
'''Implementation of the models\' parameters.'''

import torch
from ..priors import ExpFamilyPrior


class ConstantParameter:
    'Simple wrapper over ``torch.Tensor`` to handle fixed parameters.'

    __repr_str = '{classname}(value={value})'

    def __init__(self, tensor, fixed_dtype=False):
        self.fixed_dtype = fixed_dtype
        self.value = tensor

    def __repr__(self):
        return self.__repr_str.format(classname=self.__class__.__name__,
                                      value=self.value)

    def __hash__(self):
        return hash(super().__repr__())

    def float_(self):
        'Convert value of the parameter to float precision.'
        if not self.fixed_dtype:
            self.value = self.value.float()

    def double_(self):
        'Convert the value of the parameter to double precision.'
        if not self.fixed_dtype:
            self.value = self.value.double()

    def to_(self, device):
        '''Move the internal buffer of the parameter to the given
        device.

        Parameters:
            device (``torch.device``): Device on which to move on

        '''
        self.value = self.value.to(device)

class BayesianParameter:
    '''Parameter which has a *prior* and a *posterior* distribution.

    Note:
        This class is hashable and therefore can be used as a key in a
        dictionary.

    Attributes:
        natural_grad (``torch.Tensor``): Natural gradient of the ELBO
            w.r.t. to the hyper-parameters of the posterior
            distribution.
        prior (:any:`beer.ExpFamilyPrior`): Prior distribution over the
            parameter.
        posterior (:any:`beer.ExpFamilyPrior`): Posterior distribution
            over the parameter.
    '''
    __repr_str = 'BayesianParameter(prior={prior}, posterior={posterior})'


    def __init__(self, prior, posterior):
        self._callbacks = set()
        self.prior, self.posterior = prior, posterior
        dtype = self.prior.natural_parameters.dtype
        device = self.prior.natural_parameters.device
        self.natural_grad = \
            torch.zeros_like(self.prior.natural_parameters, dtype=dtype,
                            device=device, requires_grad=False)

    def __repr__(self):
        return self.__repr_str.format(prior=self.prior, posterior=self.posterior)

    def __hash__(self):
        return hash(super().__repr__())

    def _dispatch(self):
        for callback in self._callbacks:
            callback()

    def register_callback(self, callback):
        '''Register a callback function that will be called every time
        the parameters if updated.

        Args:
            callback (fucntion): Function to call.

        '''
        self._callbacks.add(callback)

    def expected_value(self):
        '''Expected value of the parameter w.r.t. the posterior
        distribution of the parameter.

        Returns:
            ``torch.Tensor``
        '''
        return self.posterior.expected_value()

    def expected_natural_parameters(self):
        '''Expected value of the natural form of the parameter w.r.t.
        the posterior distribution of the parameter.

        Returns:
            ``torch.Tensor``
        '''
        return self.posterior.expected_sufficient_statistics()

    def accumulate_natural_grad(self, acc_stats):
        '''Accumulate the natural gradient from the accumulated
        statistics.

        Args:
            acc_stats (``torch.Tensor[dim]``): Accumulated statistics
                of the parameter.

        '''
        natural_grad = self.prior.natural_parameters + acc_stats - \
            self.posterior.natural_parameters
        self.natural_grad += natural_grad.detach()

    def natural_grad_update(self, lrate):
        self.posterior.natural_parameters = torch.tensor(
            self.posterior.natural_parameters + \
            lrate * self.natural_grad,
            requires_grad=True
        )
        # Notify the observers the parameters has changed.
        self._dispatch()

    def kl_div(self):
        '''KL divergence posterior/prior.'''
        return ExpFamilyPrior.kl_div(self.posterior, self.prior)

    def float_(self):
        '''Convert value of the parameter to float precision.'''
        self.prior = self.prior.float()
        self.posterior = self.posterior.float()
        self.natural_grad = self.natural_grad.float()

    def double_(self):
        '''Convert the value of the parameter to double precision.'''
        self.prior = self.prior.double()
        self.posterior = self.posterior.double()
        self.natural_grad = self.natural_grad.double()

    def to_(self, device):
        '''Move the internal buffer of the parameter to the given
        device.

        Parameters:
            device (``torch.device``): Device on which to move on

        '''
        self.prior = self.prior.to(device)
        self.posterior = self.posterior.to(device)
        self.natural_grad = self.natural_grad.to(device)


class BayesianParameterSet:
    '''Set of Bayesian parameters.'''

    def __init__(self, parameters):
        self.__parameters = parameters

    def __len__(self):
        return len(self.__parameters)

    def __getitem__(self, key):
        return self.__parameters[key]

    def expected_natural_parameters(self):
        '''Expected value of the natural form of the parameters w.r.t.
        their posterior distribution.

        Returns:
            ``torch.Tensor[k,dim`` where k is the number of elements of
                the set.
        '''
        return torch.cat([param.expected_natural_parameters().view(1, -1)
                          for param in self.__parameters], dim=0)

    def float_(self):
        '''Convert value of the parameter to float precision in-place.'''
        for param in self.__parameters:
            param.float()

    def double_(self):
        '''Convert the value of the parameter to double precision
        in-place.'''
        for param in self.__parameters:
            param.double()

    def to_(self, device):
        '''Move the internal buffer of the parameter to the given
        device in-place.

        Parameters:
            device (``torch.device``): Device on which to move on

        '''
        for param in self.__parameters:
            param.to(device)


__all__ = [
    'ConstantParameter',
    'BayesianParameter',
    'BayesianParameterSet'
]
