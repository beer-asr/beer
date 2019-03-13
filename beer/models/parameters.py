import abc
import uuid
import typing
import torch
from ..dists import ExponentialFamily
from ..dists import kl_div


__all__ = ['BayesianParameter', 'BayesianParameterSet',
           'ConjugateBayesianParameter', 'NonConjugateBayesianParameter']


class BayesianParameter(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Base class for a Bayesian Parameter (i.e. a parameter with a prior
     and a posterior distribution.

    '''

    def __init__(self, init_stats, prior, posterior=None, likelihood_fn=None):
        super().__init__()
        self.prior = prior
        self.posterior = posterior
        self.likelihood_fn = likelihood_fn
        self.register_buffer('stats', init_stats.clone().detach())
        self._callbacks = set()
        self._uuid = uuid.uuid4()

    # We override the default repr provided by torch's modules to make
    # the BEER model tree clearer.
    def __repr__(self):
        class_name = self.__class__.__qualname__
        prior_name = self.prior.__class__.__qualname__
        if self.posterior is not None:
            post_name = self.posterior.__class__.__qualname__
        else:
            post_name = '<unspecified>'
        return f'{class_name}(prior={prior_name}, posterior={post_name})'

    def __hash__(self):
        return hash(self._uuid)

    def __eq__(self, other):
        if other.__class__ is other.__class__:
            return hash(self) == hash(other)
        raise NotImplementedError

    def dispatch(self):
        'Notify the observers the parameter has changed.'
        for callback in self._callbacks:
            callback()

    def register_callback(self, callback):
        '''Register a callback function that will be called every time
        the parameters if updated. The function takes no argument.

        '''
        self._callbacks.add(callback)

    def zero_stats(self):
        'Reset the accumulated statistics.'
        self.stats.zero_()

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

    ####################################################################
    # TODO: to be removed

    def expected_natural_parameters(self):
        import warnings
        warnings.warn('The "expected_natural_parameters" method is ' \
                      'deprecated. Use the "natural_form" method instead.',
                      DeprecationWarning, stacklevel=2)
        return self.natural_form()

    def expected_value(self):
        import warnings
        warnings.warn('The "expected_value" method is ' \
                      'deprecated. Use the "value" method instead.',
                      DeprecationWarning, stacklevel=2)
        return self.value()

    ####################################################################
    # Bayesian parameters is iterable as it can represent a set of
    # parameters.

    def __len__(self):
        if len(self.stats.shape) <= 1:
            return 1
        return self.stats.shape[0]

    def __getitem__(self, key):
        return self.__class__(prior=self.prior[key],
                              posterior=self.posterior[key],
                              init_stats=self.stats[key])

    ####################################################################
    # Interface to be implemented by other subclasses.

    @abc.abstractmethod
    def value(self):
        '''Value of the parameter w.r.t. the posterior
        distribution of the parameter. Note that, according to the
        concrete class of the parameter, the "type" of the
        returned value depends on the concrete paramter class. For
        instance, it can be the expectation of the natural form of the
        parameter w.r.t. the posterior distribution or a stochastic
        sampled from the posterior distribution.

        Returns:
            ``torch.Tensor`` or eventually a tuple of ``torch.Tensor``.
        '''
        pass

    @abc.abstractmethod
    def natural_form(self):
        '''Natural form of the parameter. Note that, according to the
        concrete class of the parameter, the "type" of the
        returned value may vary. For instance, it can be the expectation
        or a sampled drawn from the posterior distribution.

        Returns:
            ``torch.Tensor``.
        '''
        pass

    @abc.abstractmethod
    def kl_div_posterior_prior(self):
        '''KL divergence between the posterior and the prior.'''
        pass


class BayesianParameterSet(torch.nn.ModuleList):
    '''Set of Bayesian parameters.'''

    def expected_natural_parameters(self):
        import warnings
        warnings.warn('The "expected_natural_parameters" method is ' \
                      'deprecated. Use the "natural_form" method instead.',
                       DeprecationWarning, stacklevel=2)
        return self.natural_form()

    def natural_form(self):
        '''Natural form of the parameters.

        Returns:
            ``torch.Tensor[k,dim]`` where k is the number of elements of
                the set.
        '''
        return torch.cat([
            param.natural_form().view(1, -1)
            for param in self],
        dim=0)


class ConjugateBayesianParameter(BayesianParameter):
    '''Parameter for model having likelihood conjugate to its prior.

    Note:
        The type of the prior is the same as the posterior.
    '''

    def __init__(self, prior, posterior, init_stats=None,
                 likelihood_fn=None):
        if init_stats is None:
            init_stats = torch.zeros_like(prior.natural_parameters())
        lhf = likelihood_fn if likelihood_fn is not None else prior.conjugate()
        super().__init__(init_stats, prior, posterior, lhf)

    def value(self):
        return self.posterior.expected_value()

    def natural_form(self):
        return self.posterior.expected_sufficient_statistics()

    def kl_div_posterior_prior(self):
        return kl_div(self.posterior, self.prior)

    def natural_grad_update(self, lrate):
        prior_nparams = self.prior.natural_parameters()
        posterior_nparams = self.posterior.natural_parameters()
        natural_grad = prior_nparams + self.stats - posterior_nparams
        new_nparams = posterior_nparams + lrate * natural_grad
        self.posterior.update_from_natural_parameters(new_nparams)
        self.dispatch()


class NonConjugateBayesianParameter(BayesianParameter):
    'Parameter for model having a posterior/prior.'

    def __init__(self, prior, posterior, init_stats=None,
                 likelihood_fn=None):
        if init_stats is None:
            init_stats = torch.zeros_like(prior.natural_parameters())
        super().__init__(init_stats, prior, posterior)

    def value(self):
        return self.posterior.expected_value()

    def natural_form(self):
        return self.posterior.expected_sufficient_statistics()

    def kl_div_posterior_prior(self):
        return kl_div(self.posterior, self.prior)

