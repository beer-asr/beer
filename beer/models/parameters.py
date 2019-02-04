from dataclasses import dataclass
import typing
import torch
from ..priors import ExpFamilyPrior
from ..dists import ExponentialFamily

@dataclass(init=False)
class BayesianParameter(torch.nn.Module):
    'Parameter which has a *prior* and a *posterior* distribution.'

    prior: ExponentialFamily
    posterior: ExponentialFamily
    stats: torch.Tensor
    _callbacks: typing.Set

    def __init__(self, prior, posterior):
        super().__init__()
        self.prior = prior
        self.posterior = posterior
        self.stats = None
        self._callbacks = set()

    def __hash__(self):
        return hash(id(self))

    def _dispatch(self):
        for callback in self._callbacks:
            callback()

    def register_callback(self, callback):
        '''Register a callback function that will be called every time
        the parameters if updated. The function takes no argument.

        '''
        self._callbacks.add(callback)

    def expected_value(self):
        '''Expected value of the parameter w.r.t. the posterior
        distribution of the parameter.

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
        # To avoid memory issue, we make sure that the stored
        # statistics are not differentiable (therefore they do not keep
        # track of the computation graph).
        if acc_stats.requires_grad:
            self.stats = acc_stats.clone().detach()
        else:
            self.stats = acc_stats

    def natural_grad_update(self, lrate):
        prior_nparams = self.prior.natural_parameters()
        posterior_nparams = self.posterior.natural_parameters()
        natural_grad = prior_nparams + self.stats - posterior_nparams
        new_nparams = posterior_nparams + lrate * natural_grad
        self.posterior.update_from_natural_parameters(new_nparams)

        # Notify the observers the parameter has changed.
        self._dispatch()


class BayesianParameterSet(torch.nn.Module):
    '''Set of Bayesian parameters.'''

    def __init__(self, parameters):
        super().__init__()
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
