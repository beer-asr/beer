
'Bayesian Mixture model.'

from itertools import chain
from .model import ConjugateExponentialModel
from ..expfamily import DirichletPrior, kl_div
import math
import torch
import torch.autograd as ta


class Mixture(ConjugateExponentialModel):
    'Bayesian Mixture Model.'

    @staticmethod
    def create(n_components, create_component_func, args={}, prior_count=1):
        '''Create a Bayesian Mixture model.

        Args:
            n_components (int): number of components in the mixture.
            create_component_func (function): function to create the
                mixture components.
            args (dictionary): arguments to pass to \
                ``create_component_func``
            prior_count (float): Strength of the prior over the weights.

        Returns:
            ``Mixture``: An initialized Mixture model.

        '''
        # Create the prior/posterior over the weights of the mixture.
        prior_weights = DirichletPrior(torch.ones(n_components) * prior_count)
        posterior_weights = DirichletPrior(torch.ones(n_components) * prior_count)

        # Create the components of the mixture.
        components = [create_component_func(**args)
                      for i in range(n_components)]

        return Mixture(prior_weights, components, posterior_weights)

    def __init__(self, prior_weights, components, posterior_weights):
        # This will be initialize in the _prepare() call.
        self._np_params_matrix = None
        self.prior_weights = prior_weights
        self.components = components
        self.posterior_weights = posterior_weights
        self._prepare()

    @property
    def weights(self):
        'Expected value of the weights.'
        w = torch.exp(self.posterior_weights.expected_sufficient_statistics)
        return w / w.sum()

    def sufficient_statistics(self, X):
        """Compute the sufficient statistics of the data.

        Args:
            X (numpy.ndarray): Data.

        Returns:
            (numpy.ndarray): Sufficient statistics of the data.

        """
        return torch.cat([self.components[0].sufficient_statistics(X),
                     torch.ones(X.size(0))[:, None]], dim=-1)

    def _prepare(self):
        matrix = torch.cat([component.posterior.expected_sufficient_statistics[None]
            for component in self.components], dim=0)
        self._np_params_matrix = torch.cat([matrix,
            self.posterior_weights.expected_sufficient_statistics[:, None]], dim=1)

    def expected_natural_params(self, mean, var):
        # TODO: pytorch version
        '''Expected value of the natural parameters of the model given
        the sufficient statistics.

        '''
        T = self.components[0].sufficient_statistics_from_mean_var(mean, var)
        T2 = np.c_[T, np.ones(T.shape[0])]

        # Inference.
        per_component_exp_llh = T2 @ self._np_params_matrix.T
        exp_llh = logsumexp(per_component_exp_llh, axis=1)
        resps = np.exp(per_component_exp_llh - exp_llh[:, None])

        # Build the matrix of expected natural parameters.
        matrix = np.c_[[component.expected_natural_params(mean, var)[0][0]
                        for component in self.components]]

        # Accumulate the sufficient statistics.
        acc_stats = resps.T @ T2[:, :-1], resps.sum(axis=0)

        return (resps @ matrix), acc_stats


    def exp_llh(self, X, accumulate=False):
        '''Expected value of the log-likelihood w.r.t to the posterior
        distribution over the parameters.

        Args:
            X (Tensor): Data as a matrix.
            accumulate (boolean): If True, returns the accumulated
                statistics.

        Returns:
            Tensor: Per-frame expected value of the log-likelihood.
            tuple(Tensor, Tensor): Accumulated statistics
                (if ``accumulate=True``).

        '''
        T = self.sufficient_statistics(X)

        # Note: the lognormalizer is already included in the expected
        # value of the natural parameters.
        per_component_exp_llh = T @ self._np_params_matrix.t()

        # Components' responsibilities.
        # The following line is equivatent to:
        #    >>> scipy.special.logsumexp(per_component_exp_llh, axis=1)
        s, _ = torch.max(per_component_exp_llh, dim=1, keepdim=True)
        exp_llh = s + (per_component_exp_llh - s).exp().sum(dim=1, keepdim=True).log()

        resps = torch.exp(per_component_exp_llh - exp_llh)

        # Add the log base measure.
        exp_llh -= .5 * X.size(1) * math.log(2 * math.pi)

        if accumulate:
            acc_stats = resps.t() @ T[:, :-1], resps.sum(dim=0)
            return exp_llh, acc_stats

        return exp_llh

    def kl_div_posterior_prior(self):
        '''KL divergence between the posterior and prior distribution.

        Returns:
            float: KL divergence.

        '''
        retval = kl_div(self.posterior_weights, self.prior_weights)
        for component in self.components:
            retval += kl_div(component.posterior, component.prior)
        return retval

    def natural_grad_update(self, acc_stats, scale, lrate):
        '''Perform a natural gradient update of the posteriors'
        parameters.

        Args:
            acc_stats (dict): Accumulated statistics.
            scale (float): Scale of the sufficient statistics.
            lrate (float): Learning rate.

        '''
        comp_stats, weights_stats = acc_stats

        # Update the components.
        for i, component in enumerate(self.components):
            component.natural_grad_update(comp_stats[i], scale, lrate)

        # Update the weights.
        natural_grad = self.prior_weights.natural_params \
            + scale * weights_stats - self.posterior_weights.natural_params
        self.posterior_weights.natural_params = ta.Variable(\
            self.posterior_weights.natural_params + lrate * natural_grad,
            requires_grad=True)

        self._prepare()

    def split(self):
        '''Split each component into two sub-components.

        Returns:
            ``Mixture``: A new mixture with two times more
                components.

        '''
        # Create the prior/posterior over the weights.
        prior_np = .5 * np.c_[self.prior_weights.natural_params,
                              self.prior_weights.natural_params].ravel()
        post_np = .5 * np.c_[self.posterior_weights.natural_params,
                             self.posterior_weights.natural_params].ravel()
        new_prior_weights = DirichletPrior(prior_np)
        new_posterior_weights = DirichletPrior(post_np)

        # Split the Normal distributions.
        new_components = [comp.split() for comp in self.components]
        new_components = list(chain.from_iterable(new_components))

        return Mixture(new_prior_weights, new_components,
                       new_posterior_weights)

