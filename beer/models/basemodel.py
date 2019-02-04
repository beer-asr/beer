import abc
import torch
from ..dists import kl_div

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

    def bayesian_parameters(self):
        'Return an iterator over the Bayesian parameters of the model.'
        for group in self.mean_field_factorization():
            for param in group:
                yield param

    def kl_div_posterior_prior(self):
        '''Kullback-Leibler divergence between the posterior/prior
        distribution of the "global" parameters.

        Returns:
            float: KL( q || p)

        '''
        return sum([kl_div(param.posterior, param.prior)
                    for param in self.bayesian_parameters()])

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

