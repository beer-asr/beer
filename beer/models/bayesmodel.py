
'Abstract Base Class for all "standard" Bayesian models.'

import abc
import torch

from ..expfamilyprior import ExpFamilyPrior


_BAESIAN_PARAMETER_REPR_STRING = 'BayesianParameter(prior_type={type})'

_BAESIAN_MODEL_REPR_STRING = \
'''{name}:
  {bayesian_parameters}

'''

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
        self._bayesian_parameters = {}
        self.__nnet_parameters = {}
        self.__const_parameters = {}
        self.__cache = {}

    def __setattr__(self, name, value):
        if isinstance(value, BayesianParameter) or isinstance(value,
                                                        BayesianParameterSet):
            self._bayesian_parameters[name] = value
        super().__setattr__(name, value)

    def __call__(self, data, **kwargs):
        return self.forward(data, **kwargs)

    def _to_string(self, indent_level=0):
        indentation = ' ' * indent_level
        retval = indentation + self.__class__.__name__
        retval += '\n' + indentation
        indentation += indentation + '  '
        for name, param in self._bayesian_parameters.items():
            retval += indentation + '(' + name + '):'
            if isinstance(param, BayesianParameter):
                retval += param._to_string(1)
            else:
                retval += '\n'
                retval += param._to_string(len(indentation) + 2)

        return retval

    def __repr__(self):
        return self._to_string()

    @property
    def grouped_parameters(self):
        '''All the Bayes parameters of the model organized into groups
        to be optimized with a coordinate ascent algorithm.

        '''
        return self.mean_field_factorization()

    @property
    def cache(self):
        '''Dictionary object used to store intermediary results while
        computing the ELBO.

        '''
        return self.__cache

    def clear_cache(self):
        '''Clear the cache.'''
        self.__cache = {}

    def kl_div_posterior_prior(self):
        '''Kullback-Leibler divergence between the posterior/prior
        distribution of the "global" parameters.

        Returns:
            float: KL( q || p)

        '''
        retval = 0.
        for parameter in self.parameters:
            retval += parameter.kl_div
        return retval

    def float(self):
        '''Create a new :any:`BayesianModel` with all the parameters set
        to float precision.

        Returns:
            :any:`BayesianModel`

        '''
        pass

    def double(self):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Create a new :any:`BayesianModel` with all the parameters set to
        double precision.

        Returns:
            :any:`BayesianModel`

        '''
        pass

    def to(self, device):
        '''Create a new :any:`BayesianModel` with all the parameters
        allocated on `device`.

        Returns:
            :any:`BayesianModel`

        '''
        pass

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
        self._modelset = modelset

    @property
    def modelset(self):
        '''Density for each value of the discrete latent variable.'''
        return self._modelset

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
    'BayesianParameter',
    'BayesianParameterSet'
]
