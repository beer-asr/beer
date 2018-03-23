
'Bayesian Mixture model.'

from itertools import chain
from .model import ConjugateExponentialModel
from ..expfamily import DirichletPrior, kl_div
import math
import torch
import torch.autograd as ta



def _logsumexp(tensor):
    'Equivatent to: scipy.special.logsumexp(tensor, axis=1)'
    s, _ = torch.max(tensor, dim=1, keepdim=True)
    return s + (tensor - s).exp().sum(dim=1, keepdim=True).log()


class Mixture(ConjugateExponentialModel):
    'Bayesian Mixture Model.'

    @staticmethod
    def _expand_labels(labels, ncomp):
        retval = torch.zeros(len(labels), ncomp)
        idxs = torch.range(0, len(labels) - 1).long()
        retval[idxs, labels] = 1
        return retval

    @staticmethod
    def create(prior_counts, create_component_func, args={}):
        '''Create a Bayesian Mixture model.

        Args:
            prior_count (Tensor): Prior count for each class.
            create_component_func (function): function to create the
                mixture components.
            args (dictionary): arguments to pass to \
                ``create_component_func``

        Returns:
            ``Mixture``: An initialized Mixture model.

        '''
        n_components = len(prior_counts)

        # Create the prior/posterior over the weights of the mixture.
        prior_weights = DirichletPrior(prior_counts)
        posterior_weights = DirichletPrior(prior_counts)

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
    def expected_comp_params(self):
        return self._np_params_matrix

    @property
    def weights(self):
        'Expected value of the weights.'
        w = torch.exp(self.posterior_weights.expected_sufficient_statistics)
        return w / w.sum()

    def sufficient_statistics(self, X):
        '''Compute the sufficient statistics of the data.

        Args:
            X (Tensor): Data.

        Returns:
            (Tensor): Sufficient statistics of the data.

        '''
        ones = torch.ones(X.size(0)).type(X.type())
        return torch.cat([self.components[0].sufficient_statistics(X),
                     ones[:, None]], dim=-1)

    def sufficient_statistics_from_mean_var(self, means, vars):
        '''Compute the sufficient statistics of the data.

        Args:
            means (Tensor): Mean for each data point.
            vars (Tensor): Variance for each data point.

        Returns:
            (Tensor): Sufficient statistics of the data.

        '''
        ones = torch.ones(means.size(0)).type(means.type())
        s_stats = self.components[0].sufficient_statistics_from_mean_var(
            means, vars)
        return torch.cat([s_stats, ones[:, None]], dim=-1)

    def _prepare(self):
        matrix = torch.cat([component.posterior.expected_sufficient_statistics[None]
            for component in self.components], dim=0)
        self._np_params_matrix = torch.cat([matrix,
            self.posterior_weights.expected_sufficient_statistics[:, None]], dim=1)

    def expected_natural_params(self, mean, var, labels=None):
        '''Expected value of the natural parameters of the model given
        the sufficient statistics.

        '''
        T = self.components[0].sufficient_statistics_from_mean_var(mean, var)
        T2 = torch.cat([T, torch.ones(T.size(0), 1).type(mean.type())], dim=-1)

        # Inference.
        per_component_exp_llh = T2 @ self._np_params_matrix.t()
        exp_llh = _logsumexp(per_component_exp_llh)
        if labels is None:
            resps = torch.exp(per_component_exp_llh - exp_llh.view(-1, 1))
        else:
            resps = self._expand_labels(labels,
                len(self.components)).type(mean.type())

        # Build the matrix of expected natural parameters.
        matrix = torch.cat([component.expected_natural_params(mean, var)[0]
            for component in self.components], dim=0)

        # Accumulate the sufficient statistics.
        acc_stats = resps.t() @ T2[:, :-1], resps.sum(dim=0)

        return (resps @ matrix), acc_stats

    def predictions_from_mean_var(self, means, vars):
        '''Per-frame probability of the compoents given a Normal
        distribution for each frame.

        Args:
            means (Tensor): Mean for each frame.
            vars (Tensor): Variance for each frame.

        Returns:
            (Tensor): Per-frame probability.

        '''
        T = self.sufficient_statistics_from_mean_var(means, vars)
        per_component_exp_llh = T @ self._np_params_matrix.t()
        exp_llh = _logsumexp(per_component_exp_llh)
        return torch.exp(per_component_exp_llh - exp_llh)

    def predictions(self, X):
        '''Per frame probability of the components given the data.

        Args:
            X (Tensor): The data.

        Returns:
            Tensor: Per-frame probability.

        '''
        T = self.sufficient_statistics(X)
        per_component_exp_llh = T @ self._np_params_matrix.t()
        exp_llh = _logsumexp(per_component_exp_llh)
        return torch.exp(per_component_exp_llh - exp_llh)

    def exp_llh(self, X, accumulate=False, labels=None):
        '''Expected value of the log-likelihood w.r.t to the posterior
        distribution over the parameters.

        Args:
            X (Tensor): Data as a matrix.
            accumulate (boolean): If True, returns the accumulated
                statistics.
            labels (Tensor): The labels alignments.

        Returns:
            Tensor: Per-frame expected value of the log-likelihood.
            tuple(Tensor, Tensor): Accumulated statistics
                (if ``accumulate=True``).

        '''
        T = self.sufficient_statistics(X)
        per_component_exp_llh = T @ self._np_params_matrix.t()
        exp_llh = _logsumexp(per_component_exp_llh)
        if labels is None:
            resps = torch.exp(per_component_exp_llh - exp_llh.view(-1, 1))
        else:
            resps = self._expand_labels(labels,
                len(self.components)).type(X.type())

        # Add the log base measure.
        exp_llh -= .5 * X.size(1) * math.log(2 * math.pi)

        # Make sure it is a single dimension vector.
        exp_llh = exp_llh.view(-1)

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

