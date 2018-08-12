
'Abstract Base Class for all "standard" Bayesian models.'

import abc
import torch

from .parameters import BayesianParameter
from .parameters import BayesianParameterSet



class BayesianModel(metaclass=abc.ABCMeta):
    '''Abstract base class for all the models.

    Attributes:
        parameters (list): List of :any:`BayesianParameter` that the
            model has registered.

    Note:
        All the classes that inherits from :any:`BayesianModel`  are
        callable, i.e.:

        .. code-block:: python

           llh = model(some_data)

        Calling a model will by default call the :any:`forward` method
        of the object and return the expected log-likelihood of the data
        given the model.
    '''

    def __init__(self):
        self._submodels = {}
        self._bayesian_parameters = {}
        self._modules = {}
        self._const_parameters = {}
        self._cache = {}

    def _register_submodel(self, name, submodel):
        if hasattr(self, name):
            self._unregister_submodel(name)
        self._submodels[name] = submodel

    def _unregister_submodel(self, name):
        del self._submodels[name]

    def _register_parameter(self, name, param):
        if hasattr(self, name):
            self._unregister_parameter(name)
        self._bayesian_parameters[name] = param

    def _unregister_parameter(self, name):
        del self._bayesian_parameters[name]

    def _register_parameterset(self, name, paramset):
        if hasattr(self, name):
            self._unregister_parameterset(name)
        self._bayesian_parameters[name] = paramset

    def _unregister_parameterset(self, name):
        del self._bayesian_parameters[name]

    def _register_module(self, name, module):
        if hasattr(self, name):
            self._unregister_module(name)
        self._modules[name] = module

    def _unregister_module(self, name):
        del self._modules[name]

    def __setattr__(self, name, value):
        if isinstance(value, BayesianModel):
            self._register_submodel(name, value)
        if isinstance(value, BayesianParameter):
            self._register_parameter(name, value)
        if isinstance(value, BayesianParameterSet):
            self._register_parameterset(name, value)
        if isinstance(value, torch.nn.Module):
            self._register_module(name, value)
        super().__setattr__(name, value)

    def __call__(self, data, **kwargs):
        return self.forward(data, **kwargs)

    def __hash__(self):
        return hash(repr(self))

    @property
    def mean_field_groups(self):
        '''All the Bayes parameters of the model organized into groups
        to be optimized with a coordinate ascent algorithm.

        '''
        return self.mean_field_factorization()

    @property
    def cache(self):
        '''Dictionary object used to store intermediary results while
        computing the ELBO.

        '''
        return self._cache

    def modules_parameters(self):
        for module in self._modules.values():
            for param in module.parameters():
                yield param

    def bayesian_parameters(self):
        for param in self._bayesian_parameters.values():
            if isinstance(param, BayesianParameterSet):
                paramset = param
                for param in paramset:
                    yield param
            else:
                yield param
        for submodel in self._submodels.values():
            for param in submodel.bayesian_parameters():
                if isinstance(param, BayesianParameterSet):
                    paramset = param
                    for param in paramset:
                        yield param
                else:
                    yield param

    def clear_cache(self):
        '''Clear the cache.'''
        self._cache = {}
        for submodel in self._submodels.values():
            submodel.clear_cache()

    def kl_div_posterior_prior(self):
        '''Kullback-Leibler divergence between the posterior/prior
        distribution of the "global" parameters.

        Returns:
            float: KL( q || p)

        '''
        retval = 0.
        for parameter in self.bayesian_parameters():
            retval += parameter.kl_div().view(1)
        return retval

    def float(self):
        '''Create a new :any:`BayesianModel` with all the parameters set
        to float precision.

        Returns:
            :any:`BayesianModel`

        '''
        for parameter in self.bayesian_parameters():
            parameter.float_()
        for name, module in self._modules.items():
            setattr(self, name, module.float())
        return self

    def double(self):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Create a new :any:`BayesianModel` with all the parameters set to
        double precision.

        Returns:
            :any:`BayesianModel`

        '''
        for parameter in self.bayesian_parameters():
            parameter.double_()
        for name, module in self._modules.items():
            setattr(self, name, module.double())
        return self

    def to(self, device):
        '''Create a new :any:`BayesianModel` with all the parameters
        allocated on `device`.

        Returns:
            :any:`BayesianModel`

        '''
        for parameter in self.bayesian_parameters():
            parameter.to_(device)
        for name, module in self._modules.items():
            setattr(self, name, module.to(device))
        return self

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

    def forward(self, s_stats, **kwargs):
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


class BayesianModelSet(BayesianModel, metaclass=abc.ABCMeta):
    '''Abstract base class for a set of the :any:`BayesianModel`.

    This model is used by model having discrete latent variable such
    as Mixture models  or Hidden Markov models.

    Note:
        subclasses of :any:`BayesianModelSet` are expected to be
        iterable and therefore should implement at minima:

        .. code-block:: python

           MyBayesianModelSet:

               def __getitem__(self, key):
                  ...

               def __len__(self):
                  ...

    '''

    @abc.abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class DiscreteLatentBayesianModel(BayesianModel, metaclass=abc.ABCMeta):
    '''Abstract base class for a set of :any:`BayesianModel` with
    discrete latent variable.

    '''

    def __init__(self, modelset):
        super().__init__()
        self.modelset = modelset

    def posteriors(self, data, **kwargs):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModelSet`.

        Compute the probability of the discrete latent variable given
        the data.

        Args:
            ``torch.Tensor[nframes, d]``: Data as a tensor.
            kwargs: model specific arguments.

        Returns:
            ``torch.Tensor[nframes, ncomp]``

        '''
        pass


__all__ = [
    'BayesianModel',
    'BayesianModelSet',
    'DiscreteLatentBayesianModel',
]
