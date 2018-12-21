
'''Implementation of the models\' parameters.'''

import uuid
import torch
from ..priors import ExpFamilyPrior


class ConstantParameter:
    'Simple wrapper over ``torch.Tensor`` to handle fixed parameters.'

    __repr_str = '{classname}(value={value})'

    def __init__(self, tensor, fixed_dtype=False):
        self.fixed_dtype = fixed_dtype
        self.value = tensor
        self.uuid = uuid.uuid4()

    def __repr__(self):
        return self.__repr_str.format(classname=self.__class__.__name__,
                                      value=self.value)

    def __hash__(self):
        return hash(self.uuid)

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
        self.stats = \
            torch.zeros_like(self.prior.natural_parameters, dtype=dtype,
                            device=device, requires_grad=False)
        self.uuid = uuid.uuid4()

    def __getstate__(self):
        self.stats = torch.tensor(self.stats)
        return self.__dict__

    def __repr__(self):
        return self.__repr_str.format(prior=self.prior, posterior=self.posterior)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        return hash(self) == hash(other)

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

    def store_stats(self, acc_stats):
        '''Store the accumulated statistics.

        Args:
            acc_stats (``torch.Tensor[dim]``): Accumulated statistics
                of the parameter.

        '''
        self.stats = acc_stats

    def remove_stats(self, acc_stats):
        self.posterior.natural_parameters = self.posterior.natural_parameters - acc_stats

    def add_stats(self, acc_stats):
        self.posterior.natural_parameters = self.posterior.natural_parameters + acc_stats

    def natural_grad_update(self, lrate):
        grad = self.prior.natural_parameters + self.stats - \
               self.posterior.natural_parameters
        self.posterior.natural_parameters = self.posterior.natural_parameters \
                                            + lrate * grad

        # Notify the observers the parameters has changed.
        self._dispatch()

    def kl_div(self):
        '''KL divergence posterior/prior.'''
        return ExpFamilyPrior.kl_div(self.posterior, self.prior)

    def float_(self):
        '''Convert value of the parameter to float precision.'''
        self.prior = self.prior.float()
        self.posterior = self.posterior.float()
        self.stats = self.stats.float()

    def double_(self):
        '''Convert the value of the parameter to double precision.'''
        self.prior = self.prior.double()
        self.posterior = self.posterior.double()
        self.stats = self.stats.double()

    def to_(self, device):
        '''Move the internal buffer of the parameter to the given
        device.

        Parameters:
            device (``torch.device``): Device on which to move on

        '''
        self.prior = self.prior.to(device)
        self.posterior = self.posterior.to(device)
        self.stats = self.stats.to(device)


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
            param.float_()

    def double_(self):
        '''Convert the value of the parameter to double precision
        in-place.'''
        for param in self.__parameters:
            param.double_()

    def to_(self, device):
        '''Move the internal buffer of the parameter to the given
        device in-place.

        Parameters:
            device (``torch.device``): Device on which to move on

        '''
        for param in self.__parameters:
            param.to_(device)


__all__ = [
    'ConstantParameter',
    'BayesianParameter',
    'BayesianParameterSet'
]
