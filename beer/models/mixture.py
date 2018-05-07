
'Bayesian Mixture model.'

import torch
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianParameter


def _expand_labels(labels, ncomp):
    retval = torch.zeros(len(labels), ncomp)
    idxs = torch.range(0, len(labels) - 1).long()
    retval[idxs, labels] = 1
    return retval


def _logsumexp(tensor):
    'Equivatent to: scipy.special.logsumexp(tensor, axis=1)'
    tmax, _ = torch.max(tensor, dim=1, keepdim=True)
    return tmax + (tensor - tmax).exp().sum(dim=1, keepdim=True).log()


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
        weights = torch.exp(self.weights_params.expected_value)
        return weights / weights.sum()

    def sufficient_statistics(self, data):
        return self.components.sufficient_statistics(data)

    # pylint: disable=C0103
    # Invalid method name.
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

    def expected_natural_params(self, mean, var, labels=None, nsamples=1):
        if labels is not None:
            onehot_labels = _expand_labels(labels, len(self.components))
            self._resps = onehot_labels.type(mean.type())
        else:
            samples = mean + torch.sqrt(var) * torch.randn(nsamples,
                                                           *mean.size())
            samples = samples.view(-1, mean.size(1)).type(mean.type())
            T = self.sufficient_statistics(samples)
            per_component_exp_llh = self.components(T)
            per_component_exp_llh += \
                self.weights_params.expected_value.view(1, -1)
            exp_llh = _logsumexp(per_component_exp_llh).view(-1)
            self._resps = torch.exp(per_component_exp_llh - exp_llh.view(-1, 1))
            self._resps = self._resps.view(nsamples, mean.size(0),
                                           len(self.components)).mean(dim=0)
        matrix = self.components.expected_natural_params_as_matrix()
        return self._resps @ matrix

    def forward(self, s_stats, labels=None):
        per_component_exp_llh = self.components(s_stats)
        per_component_exp_llh += self.weights_params.expected_value.view(1, -1)
        if labels is not None:
            onehot_labels = _expand_labels(labels, len(self.components))
            onehot_labels = onehot_labels.type(s_stats.type())
            exp_llh = (per_component_exp_llh * onehot_labels).sum(dim=-1)
            self._resps = onehot_labels
        else:
            exp_llh = _logsumexp(per_component_exp_llh).view(-1)
            self._resps = torch.exp(per_component_exp_llh - exp_llh.view(-1, 1))
        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        retval = {
            self.weights_params: self._resps.sum(dim=0),
            **self.components.accumulate(s_stats, self._resps)
        }
        self._resps = None
        return retval
