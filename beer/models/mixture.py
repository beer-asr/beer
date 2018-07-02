
'Bayesian Mixture model.'

import torch
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianParameter
from ..expfamilyprior import DirichletPrior
from ..utils import onehot
from ..utils import logsumexp


class Mixture(BayesianModel):
    '''Bayesian Mixture Model.'''

    def __init__(self, prior_weights, posterior_weights, modelset):
        '''
        Args:
            prior_weights (:any:`DirichletPrior`): Prior distribution
                over the weights of the mixture.
            posterior_weights (any:`DirichletPrior`): Posterior
                distribution over the weights of the mixture.
            modelset (:any:`BayesianModelSet`): Set of models.

        '''
        super().__init__()
        self.weights_param = BayesianParameter(prior_weights, posterior_weights)
        self.modelset = modelset

    @property
    def weights(self):
        'Expected value of the weights of the mixture.'
        weights = torch.exp(self.weights_param.expected_value())
        return weights / weights.sum()

    def log_predictions(self, s_stats):
        '''Compute the log responsibility of the mixture for each frame.

        Args:
            s_stats (``torch.Tensor[n_frames, stats_dim]``): sufficient
                statistics.

        Returns:
            (``torch.Tensor[n_frames, ncomponents]``)

        '''
        per_component_exp_llh = self.modelset(s_stats)
        per_component_exp_llh += self.weights_param.expected_value().view(1, -1)
        lognorm = logsumexp(per_component_exp_llh, dim=1).view(-1)
        return per_component_exp_llh - lognorm.view(-1, 1)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @property
    def grouped_parameters(self):
        groups = self.modelset.grouped_parameters
        groups[0].insert(0, self.weights_param)
        return groups

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def float(self):
        return self.__class__(
            self.weights_param.prior.float(),
            self.weights_param.posterior.float(),
            self.modelset.float()
        )

    def double(self):
        return self.__class__(
            self.weights_param.prior.double(),
            self.weights_param.posterior.double(),
            self.modelset.double()
        )

    def to(self, device):
        return self.__class__(
            self.weights_param.prior.to(device),
            self.weights_param.posterior.to(device),
            self.modelset.to(device)
        )

    def forward(self, s_stats, labels=None):
        log_weights = self.weights_param.expected_value().view(1, -1)
        per_component_exp_llh = self.modelset(s_stats)
        per_component_exp_llh += log_weights

        if labels is not None:
            resps = onehot(labels, len(self.modelset),
                           dtype=log_weights.dtype, device=log_weights.device)
            exp_llh = (per_component_exp_llh * resps).sum(dim=-1)
            self.cache['resps'] = resps
        else:
            exp_llh = logsumexp(per_component_exp_llh, dim=1).view(-1)
            self.cache['resps'] = torch.exp(per_component_exp_llh - exp_llh.view(-1, 1))

        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        resps = self.cache['resps']
        retval = {
            self.weights_param: resps.sum(dim=0),
            **self.modelset.accumulate(s_stats, resps)
        }
        self.clear_cache()
        return retval

    def local_kl_div_posterior_prior(self, parent_msg=None):
        return self.modelset.local_kl_div_posterior_prior(self.cache['resps'])

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    def sufficient_statistics_from_mean_var(self, mean, var):
        return self.modelset.sufficient_statistics_from_mean_var(mean, var)

    def expected_natural_params(self, mean, var, labels=None,
                                nsamples=1):
        nframes = len(mean)
        ncomps = len(self.modelset)

        # Estimate the responsibilities if not given.
        if labels is not None:
            resps = onehot(labels, len(self.modelset), dtype=mean.dtype,
                           device=mean.device)
        else:
            noise =  torch.randn(nsamples, *mean.size(), dtype=mean.dtype,
                                 device=mean.device)
            samples = (mean + torch.sqrt(var) * noise).view(nframes * nsamples, -1)
            s_stats = self.sufficient_statistics(samples)
            resps = torch.exp(self.log_predictions(s_stats))
            resps = resps.view(nsamples, nframes, ncomps).mean(dim=0)

        # Store the responsibilities to accumulate the s. statistics.
        self.cache['resps'] = resps

        s_stats = self.modelset.sufficient_statistics_from_mean_var(mean, var)
        return self.modelset.expected_natural_params_from_resps(resps), s_stats


def create(model_conf, mean, variance, create_model_handle):
    dtype, device = mean.dtype, mean.device
    modelset = create_model_handle(model_conf['components'], mean, variance)
    n_element = len(modelset)
    weights = torch.ones(n_element, dtype=dtype, device=device) / n_element
    prior_strength = model_conf['prior_strength']
    prior_weights = DirichletPrior(prior_strength * weights)
    posterior_weights = DirichletPrior(prior_strength * weights)
    return Mixture(prior_weights, posterior_weights, modelset)


__all__ = ['Mixture']
