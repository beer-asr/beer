import math
import torch

from .basemodel import Model
from .parameters import ConjugateBayesianParameter
from ..dists import Gamma as GammaPdf


__all__ = ['Gamma']


########################################################################
# Helper to build the default parameters.

def _default_param(mean, prior_strength):
    shape = torch.ones_like(mean) * prior_strength
    rate = prior_strength / mean
    prior = GammaPdf.from_std_parameters(shape, rate)
    posterior = GammaPdf.from_std_parameters(shape.clone(), rate.clone())
    return ConjugateBayesianParameter(prior, posterior)

########################################################################

class Gamma(Model):
    'Gamma model with known shape parameter.'

    @classmethod
    def create(cls, mean, shape, prior_strength=1.):
        '''Create a Gamma model.

        Args:
            mean (``torch.Tensor[dim]``): Initial mean of the model.
            shape (``torch.tensor[dim]``): Fixed shape parameters.
            prior_strenght (float): Strength of the prior over the rate
                parameter.

        Returns:
            :any:`Gamma`

        '''
        tensorconf = {'dtype': mean.dtype, 'device': mean.device,
                      'requires_grad': False}
        shape = torch.tensor(shape, **tensorconf)
        mean = mean.detach()
        return cls(shape, _default_param(shape / mean, prior_strength))

    def __init__(self, shape, rate):
        super().__init__()
        self.register_buffer('shape', shape)
        self.rate = rate

    ####################################################################
    # The following properties are exposed only for plotting/debugging
    # purposes.

    @property
    def mean(self):
        return self.shape / self.rate.value()

    ####################################################################

    def sufficient_statistics(self, data):
        return self.rate.likelihood_fn.sufficient_statistics(data)

    def mean_field_factorization(self):
        return [[self.rate]]

    def expected_log_likelihood(self, stats):
        nparams = self.rate.natural_form()
        dim = nparams.shape[-1]
        rate = nparams[:dim]
        log_rate = nparams[dim:]
        lnorm = torch.lgamma(self.shape) - self.shape * log_rate
        lnorm = lnorm.sum(dim=-1, keepdims=True)
        final_nparams = torch.cat([
            -rate, 
            (self.shape - 1), 
            -lnorm,
        ], dim=-1)
        return self.rate.likelihood_fn(nparams, stats)

    def accumulate(self, stats):
        dim = stats.shape[-1] // 2
        new_stats = torch.cat([
            -stats[:, :dim],
            torch.ones_like(stats[:, :dim]) * self.shape
        ], dim=-1)
        return {self.rate: new_stats.sum(dim=0)}
