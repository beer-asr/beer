import abc
from functools import reduce
import torch
from .parameters import ConjugateBayesianParameter

__all__ = ['Model', 'DiscreteLatentModel']


class Model(torch.nn.Module, metaclass=abc.ABCMeta):
    'Abstract base class for all the models.'

    def __init__(self):
        super().__init__()
        self._cache = {}

    @property
    def cache(self):
        '''Dictionary object used to store intermediary results while
        computing the ELBO.

        '''
        return self._cache

    def clear_cache(self):
        self._cache = {}
        for module in self.modules():
            if module is not self and isinstance(module, Model):
                module.clear_cache()

    def bayesian_parameters(self, paramtype=None, paramfilter=None,
                            keepgroups=False):
        '''Return an iterator over the Bayesian parameters of the model.

        Args:
            keepgroups (boolean): If true, preserve the mean field group
                of the model.
            paramtype (class): Class of parameters to retrieve.
            paramfilter (function): function that takes a Bayesian
                parameter as argument and returns a True if the
                parameter should be returned by the iterator False
                otherwise.
        '''
        def _yield_params(group):
            for param in group:
                # Note we don't consider subclasses of paramtype as a
                # hit.
                if paramtype is None or type(param) == paramtype:
                    if paramfilter is None or paramfilter(param):
                        yield param

        for group in self.mean_field_factorization():
            if not keepgroups:
                yield from _yield_params(group)
            else:
                group = [param for param in _yield_params(group)]
                if len(group) > 0:
                    yield group

    def conjugate_bayesian_parameters(self, keepgroups=False):
        'Convenience method to retrieve the mf groups to be trained.'
        return self.bayesian_parameters(paramtype=ConjugateBayesianParameter,
                                        keepgroups=keepgroups)

    def kl_div_posterior_prior(self):
        '''Kullback-Leibler divergence between the posterior/prior
        distribution of the "global" parameters.

        Returns:
            float: KL( q || p)

        Note:
            Model that have non conjugate parameters must override this
            method.

        '''
        return sum([param.kl_div_posterior_prior().sum()
                    for param in self.bayesian_parameters()])

    def accumulated_statistics(self):
        'Accumulated statistics as a vector for all Bayesian parameters.'
        acc_stats = []
        for param in self.bayesian_parameters():
            post_nparams = param.posterior.natural_parameters()
            prior_nparams = param.prior.natural_parameters()
            acc_stats.append(post_nparams - prior_nparams)
        return torch.cat(acc_stats)

    @staticmethod
    def _replace_params(module, paramsmap):
        for name, child in module.named_children():
            if child in paramsmap:
                module.add_module(name, paramsmap[child])
            elif isinstance(child, Model):
                Model._replace_params(child, paramsmap)

    def replace_parameters(self, paramsmap):
        '''Replace the parameters of the model by new ones.

        Args:
            paramsmap (dict): Dictionary like object where the keys are
                the parameters of the model to be replaced and the
                values are the new parameters.
        '''
        Model._replace_params(self, paramsmap)

    ####################################################################
    # Abstract methods to be implemented by subclasses.
    ####################################################################

    @abc.abstractmethod
    def accumulate(self, s_stats, parent_msg=None):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Accumulate the sufficient statistics of the models necessary
        to update the parameters of the model.

        Args:
            s_stats (list): List of sufficient statistics.
            parent_msg (object): Message from the parent/co-parents
                to make the VB update.

        Returns:
            dict: Dictionary of accumulated statistics for each parameter.

        '''
        pass

    @abc.abstractmethod
    def expected_log_likelihood(self, s_stats, **kwargs):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Compute the expected log-likelihood of the data given the
        model.

        Args:
            s_stats (``torch.Tensor[n_frames, dim]``): Sufficient
                statistics of the model.
            kwargs (dict): Model specific parameters

        Returns:
            ``torch.Tensor[n_frames]``: expected log-likelihood.

        '''
        pass

    @abc.abstractmethod
    def mean_field_factorization(self):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Return the Bayesian parameters grouped into list according to
        the mean-field factorization of the VB posterior of the model.

        Returns:
            list of list of :any:`BayesianParameters`

        '''
        pass

    @abc.abstractmethod
    def sufficient_statistics(self, data):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Compute the sufficient statistics of the data.

        Args:
            data (``torch.Tensor[n_frames, dim]``): Data.

        Returns:
            (``torch.Tensor[n_frames, dim_stats]``): Sufficient \
                statistics of the data.

        '''
        pass



class DiscreteLatentModel(Model, metaclass=abc.ABCMeta):
    '''Abstract base class for a set of :any:`BayesianModel` with
    discrete latent variable.

    '''

    def __init__(self, modelset):
        super().__init__()
        self.modelset = modelset

    @abc.abstractmethod
    def posteriors(self, data, **kwargs):
        '''Compute the probability of the discrete latent variable given
        the data.

        Args:
            ``torch.Tensor[nframes, d]``: Data as a tensor.
            kwargs: model specific arguments.

        Returns:
            ``torch.Tensor[nframes, ncomp]``

        '''
        pass

