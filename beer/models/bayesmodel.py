
'Abstract Base Class for all "standard" Bayesian models.'

import abc
import torch

from .parameters import ConstantParameter
from .parameters import BayesianParameter
from .parameters import BayesianParameterSet



class BayesianModel(metaclass=abc.ABCMeta):
    '''Abstract base class for all the models.

    Attributes:
        parameters (list): List of :any:`BayesianParameter` that the
            model has registered.

    '''

    def __init__(self):
        self._submodels = {}
        self._bayesian_parameters = {}
        self._modules = {}
        self._const_parameters = {}
        self._cache = {}

    def _register_submodel(self, name, submodel):
        self._unregister_submodel(name)
        self._submodels[name] = submodel

    def _unregister_submodel(self, name):
        try:
            del self._submodels[name]
        except KeyError:
            pass

    def _register_const_parameter(self, name, param):
        self._unregister_const_parameter(name)
        self._const_parameters[name] = param

    def _unregister_const_parameter(self, name):
        try:
            del self._const_parameters[name]
        except KeyError:
            pass

    def _register_parameter(self, name, param):
        self._unregister_parameter(name)
        self._bayesian_parameters[name] = param

    def _unregister_parameter(self, name):
        try:
            del self._bayesian_parameters[name]
        except KeyError:
            pass

    def _register_parameterset(self, name, paramset):
        self._unregister_parameterset(name)
        self._bayesian_parameters[name] = paramset

    def _unregister_parameterset(self, name):
        try:
            del self._bayesian_parameters[name]
        except KeyError:
            pass

    def _register_module(self, name, module):
        self._unregister_module(name)
        self._modules[name] = module

    def _unregister_module(self, name):
        try:
            del self._modules[name]
        except KeyError:
            pass

    def __setattr__(self, name, value):
        if isinstance(value, BayesianModel):
            self._register_submodel(name, value)
        if isinstance(value, ConstantParameter):
            self._register_const_parameter(name, value)
        if isinstance(value, BayesianParameter):
            self._register_parameter(name, value)
        if isinstance(value, BayesianParameterSet):
            self._register_parameterset(name, value)
        if isinstance(value, torch.nn.Module):
            self._register_module(name, value)
        super().__setattr__(name, value)

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
        for submodel in self._submodels.values():
            for param in submodel.modules_parameters():
                yield param

    def const_parameters(self):
        for param in self._const_parameters.values():
            yield param
        for submodel in self._submodels.values():
            for param in submodel.const_parameters():
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
        for parameter in self._const_parameters.values():
            parameter.float_()
        for parameter in self._bayesian_parameters.values():
            parameter.float_()
        for name, module in self._modules.items():
            setattr(self, name, module.float())
        for name, submodel in self._submodels.items():
            setattr(self, name, submodel.float())
        return self

    def double(self):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Create a new :any:`BayesianModel` with all the parameters set to
        double precision.

        Returns:
            :any:`BayesianModel`

        '''
        for parameter in self._const_parameters.values():
            parameter.double_()
        for parameter in self._bayesian_parameters.values():
            parameter.double_()
        for name, module in self._modules.items():
            setattr(self, name, module.double())
        for name, submodel in self._submodels.items():
            setattr(self, name, submodel.double())
        return self

    def to(self, device):
        '''Create a new :any:`BayesianModel` with all the parameters
        allocated on `device`.

        Returns:
            :any:`BayesianModel`

        '''
        for parameter in self._const_parameters.values():
            parameter.to_(device)
        for parameter in self._bayesian_parameters.values():
            parameter.to_(device)
        new_modules = {}
        for name, module in self._modules.items():
            new_modules[name] = module.to(device)
        self._modules = new_modules
        new_submodels = {}
        for name, submodel in self._submodels.items():
            new_submodels[name] = submodel.to(device)
        self._submodels = new_submodels
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


class DiscreteLatentBayesianModel(BayesianModel, metaclass=abc.ABCMeta):
    '''Abstract base class for a set of :any:`BayesianModel` with
    discrete latent variable.

    '''

    def __init__(self, modelset):
        super().__init__()
        self.modelset = modelset

    @abc.abstractmethod
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
    'DiscreteLatentBayesianModel',
]

