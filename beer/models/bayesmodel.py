
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
        new_prior = prior.float()
        new_posterior = posterior.float()
        new_ngrad = self.natural_grad.float()
        new_param = BayesianParameter(new_prior, new_posterior)
        new_param.natural_grad = new_ngrad
        return new_param

    def double(self):
        '''Convert the value of the parameter to double precision.

        Returns:
            :any:`BayesianParameter`

        '''
        new_prior = prior.double()
        new_posterior = posterior.double()
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
        new_prior = prior.to(device)
        new_posterior = posterior.double(device)
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
        self._parameters = parameters

    def __len__(self):
        return len(self._parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def float(self):
        '''Convert value of the parameter to float precision.

        Returns:
            :any:`BayesianParameterSet`

        '''
        return BayesianParameterSet([
            param.float() for param in self._parameters
        ])

    def double(self):
        '''Convert the value of the parameter to double precision.

        Returns:
            :any:`BayesianParameterSet`

        '''
        return BayesianParameterSet([
            param.double() for param in self._parameters
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
            param.to(device) for param in self._parameters
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
        self._parameters = []
        self._cache = {}

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

    @property
    def parameters(self):
        '''All the :any:`BayesianParameters` of the model.'''
        return self._parameters

    @property
    def grouped_parameters(self):
        '''All the Bayes parameters of the model organized into groups
        to be optimized with a coordinate.

        Note:
            By default, for efficiency reason, all the parameters are
            put in a single group. Models which need a different
            behavior have to override this method.

        '''
        return [self._parameters]

    @property
    def cache(self):
        '''Dictionary object used to store intermediary results while
        computing the ELBO.

        '''
        return self._cache

    def clear_cache(self):
        '''Clear the cache.'''
        self._cache = {}

    def local_kl_div_posterior_prior(self, parent_msg=None):
        '''KL divergence between the posterior/prior distribution over the
        "local" parameters

        parent_msg (object): Message from the parent/co-parents
                to compute the local KL divergence.

        Returns:
            ``torch.Tensor`` or 0.
        '''
        val = self._parameters[0].expected_value()
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
           'BayesianParameterSet']
