
'Abstract Base Class for all "standard" Bayesian models.'

import abc
import torch

from ..expfamilyprior import ExpFamilyPrior


class BayesianParameter:
    '''Parameter which has a *prior* and a *posterior* distribution.

    Note:
        This class is hashable and therefore can be used as a key in a
        dictionary.

    Attributes:
        expected_value (``torch.Tensor``): Expected value of the
            parameter w.r.t. the posterior distribution.
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
        tensor_type = self.prior.natural_hparams.type()
        self.natural_grad = \
            torch.zeros_like(self.prior.natural_hparams).type(tensor_type)

    def __hash__(self):
        return hash(repr(self))

    @property
    def expected_value(self):
        return self.posterior.expected_sufficient_statistics

    def zero_natural_grad(self):
        '''Reset the natural gradient to zero.'''
        self.natural_grad.zero_()


# pylint: disable=R0903
class BayesianParameterSet:
    '''Set of Bayesian parameters.

    The purpose of this class to register list of parameters at once.

    Attributes:
        parameters (list): List of :any:`BayesianParameter`.

    '''

    def __init__(self, parameters):
        self._parameters = parameters

    def __len__(self):
        return len(self._parameters)

    def __getitem__(self, key):
        return self._parameters[key]


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
        self._parameters = []

    def __setattr__(self, name, value):
        if isinstance(value, BayesianParameter):
            self._parameters.append(value)
        elif isinstance(value, BayesianParameterSet):
            for parameter in value:
                self._parameters.append(parameter)
        elif isinstance(value, BayesianModel):
            self._parameters += value.parameters
        super().__setattr__(name, value)

    def __call__(self, data, labels=None):
        return self.forward(data, labels)

    @staticmethod
    def kl_div_posterior_prior(parameters):
        '''Kullback-Leibler divergence between the posterior and the prior
        distribution of the parameters.

        Args:
            parameters (list): List of :any:`BayesianParameter`.

        Returns:
            float: KL( q || p)

        '''
        retval = 0.
        for parameter in parameters:
            retval += ExpFamilyPrior.kl_div(parameter.posterior,
                                            parameter.prior)
        return retval

    @property
    def parameters(self):
        return self._parameters

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
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, s_stats, latent_variables=None):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Compute the Evidence Lower-BOund (ELBO) of the data given the
        model.

        Args:
            s_stats (``torch.Tensor[n_frames, dim]``): Sufficient
                statistics of the model.
            latent_variables (object): Latent variable that can be
                provided to the model (optional). Note that type of
                the latent variables depends on the model. If a model
                does not use any latent variable, it will ignore this
                parameter.

        Returns:
            ``torch.Tensor[n_frames]``: ELBO.

        '''
        raise NotImplementedError

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
        raise NotImplementedError


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
    def expected_natural_params_as_matrix(self):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Compute the expected value natural of the set parameters as
        a matrix.

        Returns:
            ``torch.Tensor``: The set of natural parameters in a matrix.
        '''
    @abc.abstractmethod
    def forward(self, s_stats, latent_variables=None):
        '''Abstract method to be implemented by subclasses of
        :any:`BayesianModel`.

        Compute the Evidence Lower-BOund (ELBO) of the data given the
        model for each model.

        Args:
            s_stats (``torch.Tensor[n_frames, dim]``): Sufficient
                statistics of the model.
            latent_variables (object): Latent variable that can be
                provided to the model (optional). Note that type of
                the latent variables depends on the model. If a model
                does not use any latent variable, it will ignore this
                parameter.

        Returns:
            ``torch.Tensor[n_frames, n_models]``: ELBO.
        '''
        raise NotImplementedError
