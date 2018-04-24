
'Bayesian Mixture model.'

import math
import torch
import torch.autograd as ta

from .bayesmodel import BayesianModel
from .bayesmodel import BayesianParameter
from ..expfamilyprior import DirichletPrior, kl_div


def _expand_labels(labels, ncomp):
    retval = torch.zeros(len(labels), ncomp)
    idxs = torch.range(0, len(labels) - 1).long()
    retval[idxs, labels] = 1
    return retval


def _logsumexp(tensor):
    'Equivatent to: scipy.special.logsumexp(tensor, axis=1)'
    s, _ = torch.max(tensor, dim=1, keepdim=True)
    return s + (tensor - s).exp().sum(dim=1, keepdim=True).log()


class Mixture(BayesianModel):
    'Bayesian Mixture Model.'

    def __init__(self, prior_weights, posterior_weights, normalset):
        '''Initialie the mixture model.

        Args:
            prior_weights (``DirichletPrior``): Prior distribution
                over the weights of the mixture.
            posterior_weights (``DirichletPrior``): Posterior
                distribution over the weights of the mixture.
            normalset (``NormalSet``): Set of normal distribution.

        '''
        super().__init__()
        self.weights_params = BayesianParameter(prior_weights, posterior_weights)
        self.components = normalset
        self._resps = None

    @property
    def weights(self):
        'Expected value of the weights.'
        w = torch.exp(self.weights_params.expected_value)
        return w / w.sum()

    def sufficient_statistics(self, X):
        return self.components.sufficient_statistics(X)

    def sufficient_statistics_from_mean_var(self, mean, var):
        return self.components.sufficient_statistics_from_mean_var(mean, var)

    def log_predictions(self, T):
        '''Compute the probability of the discrete components given the
        features.

        Args:
            T (Tensor): sufficient statistics.

        Returns:
            (Tensor): Per-frame probability of each components.

        '''
        per_component_exp_llh = self.components(T)
        per_component_exp_llh += self.weights_params.expected_value.view(1, -1)
        exp_llh = _logsumexp(per_component_exp_llh).view(-1)
        return per_component_exp_llh - exp_llh.view(-1, 1)

    def expected_natural_params(self, T):
        matrix = self.components.expected_natural_params_as_matrix()
        resps = torch.exp(self.log_predictions(T))
        return resps @ matrix

    def forward(self, T, labels=None):
        per_component_exp_llh = self.components(T)
        per_component_exp_llh += self.weights_params.expected_value.view(1, -1)
        if labels is not None:
            onehot_labels = _expand_labels(labels,
                len(self.components)).type(T.type())
            #per_component_exp_llh += torch.log(onehot_labels)
            exp_llh = (per_component_exp_llh * onehot_labels).sum(dim=-1)
            self._resps = onehot_labels
        else:
            exp_llh = _logsumexp(per_component_exp_llh).view(-1)
            self._resps = torch.exp(per_component_exp_llh - exp_llh.view(-1, 1))
        return exp_llh

    def accumulate(self, T, given_resps=None):
        if given_resps is not None:
            resps = given_resps
        else:
            resps = self._resps
        retval = {
            self.weights_params: resps.sum(dim=0),
            **self.components.accumulate(T, resps)
        }
        self._resps = None
        return retval

