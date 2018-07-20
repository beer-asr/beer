
'''Implementation of the models\' parameters.'''

import abc
import torch

from ..expfamilyprior import ExpFamilyPrior


_BAESIAN_PARAMETER_REPR_STRING = 'BayesianParameter(prior_type={type})'


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

    def __init__(self, prior, posterior):
        self.prior, self.posterior = prior, posterior
        dtype = self.prior.natural_hparams.dtype
        device = self.prior.natural_hparams.device
        self.natural_grad = \
            torch.zeros_like(self.prior.natural_hparams, dtype=dtype,
                            device=device)

    def _to_string(self, indent_level=0):
        retval = ' ' * indent_level
        retval += _BAESIAN_PARAMETER_REPR_STRING.format(
            type=repr(self.prior)
        )
        return retval

    def __repr__(self):
        return self._to_string()

    def __hash__(self):
        return hash(self)

    def expected_value(self, concatenated=True):
        '''Expected value of the sufficient statistics of the parameter
        w.r.t. the posterior distribution.

        Args:
            concatenated (boolean): If true, concatenate the sufficient
                statistics into a single ``torch.Tensor``. If false,
                the statistics are returned in a tuple.

        Returns:
            ``torch.Tensor`` or a ``tuple``
        '''
        if concatenated:
            return self.posterior.expected_sufficient_statistics
        return self.posterior.split_sufficient_statistics(
            self.posterior.expected_sufficient_statistics
        )

    def zero_natural_grad(self):
        '''Reset the natural gradient to zero.'''
        self.natural_grad.zero_()

    def kl_div(self):
        '''KL divergence posterior/prior.'''
        return ExpFamilyPrior.kl_div(self.posterior, self.prior)

    def float(self):
        '''Convert value of the parameter to float precision.'''
        self.prior = self.prior.float()
        self.posterior = self.posterior.float()
        self.natural_grad = self.natural_grad.float()
        return self

    def double(self):
        '''Convert the value of the parameter to double precision.'''
        self.prior = self.prior.double()
        self.posterior = self.posterior.double()
        self.natural_grad = self.natural_grad.double()
        return self

    def to(self, device):
        '''Move the internal buffer of the parameter to the given
        device.

        Parameters:
            device (``torch.device``): Device on which to move on

        '''
        self.prior = self.prior.to(device)
        self.posterior = self.posterior.to(device)
        self.natural_grad = self.natural_grad.to(device)
        return self


class BayesianParameterSet:
    '''Set of Bayesian parameters.'''

    def __init__(self, parameters):
        self.__parameters = parameters

    def _to_string(self, indent_level=0):
        retval = ' ' * indent_level
        for i, param in enumerate(self.__parameters):
            retval += '(' + str(i) + ') '
            retval += _BAESIAN_PARAMETER_REPR_STRING.format(
                type=repr(param.prior)
            ) + '\n' + ' ' * indent_level
        return retval

    def __repr__(self):
        return self._to_string()

    def __len__(self):
        return len(self.__parameters)

    def __getitem__(self, key):
        return self.__parameters[key]

    def float(self):
        '''Convert value of the parameter to float precision.'''
        for param in self.__parameters:
            param.float()
        return self

    def double(self):
        '''Convert the value of the parameter to double precision.'''
        for param in self.__parameters:
            param.double()
        return self

    def to(self, device):
        '''Move the internal buffer of the parameter to the given
        device.

        Parameters:
            device (``torch.device``): Device on which to move on

        '''
        for param in self.__parameters:
            param.to(device)
        return self


__all__ = [
    'BayesianParameter',
    'BayesianParameterSet'
]
