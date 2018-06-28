
'Abstract Base Class for all "standard" Bayesian models.'

import abc
import torch

from ..expfamilyprior import ExpFamilyPrior


def average_models(models, weights):
    '''Weighted average the parameters of the set of models of the same
    type.

    Args:
        models (list): Sequence of model
        weights (``torch.Tensor``): weights for each model.

    Returns:
        :any:`BayesModel`

    '''
    # Load all the models
    list_models = [model for model in models]

    # We use the first model of the list as return value.
    ret_model = list_models[0].copy()

    # Average the Bayesian parameters.
    nparams = len(list_models[0].parameters)
    for i in range(nparams):
        new_param = ret_model.parameters[i]
        new_nhparams = weights[0] * new_param.posterior.natural_hparams
        for j, model in enumerate(list_models[1:], start=1):
            new_nhparams += \
                weights[j] * model.parameters[i].posterior.natural_hparams
        new_param.posterior.natural_hparams = new_nhparams
        ret_model.parameters[i] = new_param

    # Average the non Bayesian parameters.
    nb_params = ret_model.non_bayesian_parameters()
    n_nb_params = len(nb_params)
    new_nb_params = []
    for i in range(n_nb_params):
        new_nb_param = torch.zeros_like(nb_params[i])
        for j, model in enumerate(list_models):
            nb_params = model.non_bayesian_parameters()
            new_nb_param += weights[j] * nb_params[i]
        new_nb_params.append(new_nb_param.clone())
    ret_model.set_non_bayesian_parameters(new_nb_params)
    return ret_model


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

    def __hash__(self):
        return hash(repr(self))

    def copy(self):
        '''Return a new copy of the parameter.'''
        dtype = self.prior.natural_hparams.dtype
        if dtype == torch.float32:
            return self.float()
        else:
            return self.double()

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

    def float(self):
        '''Convert value of the parameter to float precision.

        Returns:
            :any:`BayesianParameter`

        '''
        new_prior = self.prior.float()
        new_posterior = self.posterior.float()
        new_ngrad = self.natural_grad.float()
        new_param = BayesianParameter(new_prior, new_posterior)
        new_param.natural_grad = new_ngrad
        return new_param

    def double(self):
        '''Convert the value of the parameter to double precision.

        Returns:
            :any:`BayesianParameter`

        '''
        new_prior =self.prior.double()
        new_posterior = self.posterior.double()
        new_ngrad = self.natural_grad.double()
        new_param = BayesianParameter(new_prior, new_posterior)
        new_param.natural_grad = new_ngrad
        return new_param

    def to(self, device):
        '''Move the internal buffer of the parameter to the given
        device.

        Parameters:
            device (``torch.device``): Device on which to move on

        Returns:
            :any:`BayesianParameter`

        '''
        new_prior = self.prior.to(device)
        new_posterior = self.posterior.to(device)
        new_ngrad = self.natural_grad.to(device)
        new_param = BayesianParameter(new_prior, new_posterior)
        new_param.natural_grad = new_ngrad
        return new_param

class BayesianParameterSet:
    '''Set of Bayesian parameters.

    The purpose of this class to register list of parameters at once.

    Attributes:
        parameters (list): List of :any:`BayesianParameter`.

    '''

    def __init__(self, parameters):
        self.__parameters = parameters

    def __len__(self):
        return len(self.__parameters)

    def __getitem__(self, key):
        return self.__parameters[key]

    def float(self):
        '''Convert value of the parameter to float precision.

        Returns:
            :any:`BayesianParameterSet`

        '''
        return BayesianParameterSet([
            param.float() for param in self.__parameters
        ])

    def double(self):
        '''Convert the value of the parameter to double precision.

        Returns:
            :any:`BayesianParameterSet`

        '''
        return BayesianParameterSet([
            param.double() for param in self.__parameters
        ])

    def to(self, device):
        '''Move the internal buffer of the parameter to the given
        device.

        Parameters:
            device (``torch.device``): Device on which to move on

        Returns:
            :any:`BayesianParameterSet`

        '''
        return BayesianParameterSet([
            param.to(device) for param in self.__parameters
        ])


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

        Calling a model will be default call the :any:`forward` method
        of the object and return the variational lower-bound of the data
        given the model. This features is mostly to be consistent with
        ``pytorch`` models.
    '''

    def __init__(self):
        self.__parameters = []
        self.__cache = {}

    def __setattr__(self, name, value):
        if isinstance(value, BayesianParameter):
            self.__parameters.append(value)
        elif isinstance(value, BayesianParameterSet):
            for parameter in value:
                self.__parameters.append(parameter)
        elif isinstance(value, BayesianModel):
            self.__parameters += value.parameters
        super().__setattr__(name, value)

    def __call__(self, data, **kwargs):
        return self.forward(data, **kwargs)

    @property
    def parameters(self):
        '''All the :any:`BayesianParameters` of the model.'''
        return self.__parameters

    @property
    def grouped_parameters(self):
        '''All the Bayes parameters of the model organized into groups
        to be optimized with a coordinate.

        Note:
            By default, for efficiency reason, all the parameters are
            put in a single group. Models which need a different
            behavior have to override this method.

        '''
        return [self.__parameters]

    @property
    def cache(self):
        '''Dictionary object used to store intermediary results while
        computing the ELBO.

        '''
        return self.__cache

    def clear_cache(self):
        '''Clear the cache.'''
        self.__cache = {}

    def non_bayesian_parameters(self):
        '''List of all non-Bayesian parameters (i.e. parameter that
        don't have prior, posterior distribution) of the model.

        Returns:
            list of ``torch.Tensor``

        '''
        return []

    def set_non_bayesian_parameters(self, new_params):
        '''Set new values for the non Bayesian parameters.

        Args:
            new_params (list): List of ``torch.Tensor``.

        '''
        pass

    def local_kl_div_posterior_prior(self, parent_msg=None):
        '''KL divergence between the posterior/prior distribution over the
        "local" parameters

        parent_msg (object): Message from the parent/co-parents
                to compute the local KL divergence.

        Returns:
            ``torch.Tensor`` or 0.
        '''
        val = self.__parameters[0].expected_value()
        return torch.tensor(0., dtype=val.dtype, device=val.device)

    def kl_div_posterior_prior(self):
        '''Kullback-Leibler divergence between the posterior/prior
        distribution of the "global" parameters.

        Returns:
            float: KL( q || p)

        '''
        retval = 0.
        for parameter in self.parameters:
            retval += ExpFamilyPrior.kl_div(parameter.posterior,
                                            parameter.prior).view(1)
        return retval

    def copy(self):
        '''Return a new copy of the model.'''
        dtype = self.__parameters[0].prior.natural_hparams.dtype
        if dtype == torch.float32:
            return self.float()
        else:
            return self.double()

    @abc.abstractmethod
    def float(self):
        '''Create a new :any:`BayesianModel` with all the parameters set
        to float precision.

        Returns:
            :any:`BayesianModel`

        '''
        pass

    @abc.abstractmethod
    def double(self):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Create a new :any:`BayesianModel` with all the parameters set to
        double precision.

        Returns:
            :any:`BayesianModel`

        '''
        pass

    @abc.abstractmethod
    def to(self, device):
        '''Create a new :any:`BayesianModel` with all the parameters
        allocated on `device`.

        Returns:
            :any:`BayesianModel`

        '''
        pass

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

    @abc.abstractmethod
    def expected_natural_params_from_resps(self, resps):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModelSet`.

        Compute the expected value natural of the parameters as
        a vector for each frame.

        Args:
            ``torch.Tensor[nframes, nclasses]``: Per-frame
                responsibilities to averge the natural parameters of the
                set's components.

        Returns:
            ``torch.Tensor[nframes, dim]``
        '''
        pass


__all__ = ['BayesianModel', 'BayesianModelSet', 'BayesianParameter',
           'BayesianParameterSet', 'average_models']
