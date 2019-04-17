import abc
import uuid
import torch
from ..dists import ExponentialFamily
from ..dists import kl_div


__all__ = ['BayesianParameter', 'ConjugateBayesianParameter']


class BayesianParameter(torch.nn.Module):
    '''Base class for a Bayesian Parameter (i.e. a parameter with a prior
     and a posterior distribution.

    '''

    def __init__(self, prior, posterior=None):
        super().__init__()
        self.prior = prior
        self.posterior = posterior
        self.uuid = uuid.uuid4()
        self._callbacks = set()

    def __len__(self):
        return len(self.prior)

    def __getitem__(self, key):
        return self.__class__(prior=self.prior[key],
                              posterior=self.posterior[key])

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
        return hash(self.uuid)

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

    ####################################################################
    # Interface to be implemented by other subclasses.

    def value(self):
        '''Value of the parameter w.r.t. the posterior
        distribution of the parameter.

        Returns:
            ``torch.Tensor`` or eventually a tuple of ``torch.Tensor``.
        '''
        return self.posterior.expected_value()

    def kl_div_posterior_prior(self):
        return kl_div(self.posterior, self.prior)


class ConjugateBayesianParameter(BayesianParameter):
    '''Parameter for model having likelihood conjugate to its prior.

    Note:
        The type of the prior is the same as the posterior.
    '''

    def __init__(self, prior, posterior, init_stats=None,
                 likelihood_fn=None):
        super().__init__(prior, posterior)
        if init_stats is None:
            init_stats = torch.zeros_like(prior.natural_parameters())
        self.register_buffer('stats', init_stats.clone().detach())

        if likelihood_fn is None:
            likelihood_fn = prior.conjugate()
        self.likelihood_fn = likelihood_fn

    def __len__(self):
        if len(self.stats.shape) <= 1: return 1
        return self.stats.shape[0]

    def __getitem__(self, key):
        return self.__class__(prior=self.prior[key],
                              posterior=self.posterior[key],
                              init_stats=self.stats[key],
                              likelihood_fn=self.likelihood_fn)

    def zero_stats(self):
        'Reset the accumulated statistics to zero.'
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

    def natural_form(self):
        return self.posterior.expected_sufficient_statistics()

    def natural_grad_update(self, lrate):
        prior_nparams = self.prior.natural_parameters()
        posterior_nparams = self.posterior.natural_parameters()
        natural_grad = prior_nparams + self.stats - posterior_nparams
        new_nparams = posterior_nparams + lrate * natural_grad
        self.posterior.update_from_natural_parameters(new_nparams)
        self.dispatch()

