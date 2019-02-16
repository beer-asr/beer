from dataclasses import dataclass, field
import uuid
import typing
import torch
from ..dists import ExponentialFamily


__all__ = ['BayesianParameter', 'BayesianParameterSet']


@dataclass(init=False, repr=False)
class BayesianParameter(torch.nn.Module):
    'Parameter which has a *prior* and a *posterior* distribution.'

    prior: ExponentialFamily
    posterior: ExponentialFamily
    _stats: torch.Tensor = field(repr=False)
    _callbacks: typing.Set = field(repr=False)
    _uuid: uuid.UUID = field(repr=False)

    def __init__(self, prior, posterior):
        super().__init__()
        self.prior = prior
        self.posterior = posterior
        stats = torch.zeros_like(self.prior.natural_parameters())
        self.register_buffer('_stats', stats)
        self._callbacks = set()
        self._uuid = uuid.uuid4()

    # We override the default repr provided by torch's modules to make
    # the BEER model tree clearer.
    def __repr__(self):
        return '<BayesianParameter>'

    def __hash__(self):
        return hash(self._uuid)

    def __eq__(self, other):
        if other.__class__ is other.__class__:
            return hash(self) == hash(other)
        raise NotImplementedError

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

    def zero_stats(self):
        'Reset the accumulated statistics.'
        self._stats.zero_()

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
            self._stats = acc_stats.clone().detach()
        else:
            self._stats = acc_stats

    def natural_grad_update(self, lrate):
        prior_nparams = self.prior.natural_parameters()
        posterior_nparams = self.posterior.natural_parameters()
        natural_grad = prior_nparams + self._stats - posterior_nparams
        new_nparams = posterior_nparams + lrate * natural_grad
        self.posterior.update_from_natural_parameters(new_nparams)

        # Notify the observers the parameter has changed.
        self._dispatch()


@dataclass(init=False)
class BayesianParameterSet(torch.nn.Module):
    '''Set of Bayesian parameters.'''

    def __init__(self, parameters):
        super().__init__()
        self.__parameters = torch.nn.ModuleList(parameters)

    # We override the default repr provided by torch's modules to make
    # the BEER model tree clearer.
    def __repr__(self):
        return '<BayesianParameterSet>'

    def __hash__(self):
        return hash(id(self))

    def __len__(self):
        return len(self.__parameters)

    def __getitem__(self, key):
        if not isinstance(key, int):
            return self.__class__(self.__parameters[key])
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

