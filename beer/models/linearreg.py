import abc
import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel, BayesianModelSet
from ..priors import GammaPrior
from ..priors import MatrixNormalPrior
from ..utils import make_symposdef


class LinearRegression(BayesianModel):
    '''Bayesian Linear Regression.

    Attributes:
        weights: weights matrix parameter.
        precision: precision parameter.

    '''

    @classmethod
    def create(cls, weights, variance, prior_strength=1.,  weights_variance=.1,
               noise_std=.1):
        '''Create a Normal model.

        Args:
            weights (``torch.Tensor[size,dim]``): Prior weights of the model.
            variance (``torch.Tensor[1]``  or scalar):
                prior variance parameter.

        Returns:
            :any:`Normal`

        '''
        dtype, device = weights.dtype, weights.device
        cov = torch.eye(weights.shape[0], dtype=dtype, device=device)
        cov *= weights_variance / prior_strength
        prior_weights = MatrixNormalPrior(weights, cov)
        noise = torch.randn(*weights.shape, dtype=dtype, device=device)
        posterior_weights = MatrixNormalPrior(weights + noise_std * noise, cov)
        shape = torch.tensor(prior_strength, dtype=dtype, device=device)
        rate =  torch.tensor(prior_strength * variance, dtype=dtype, device=device)
        prior_precision = GammaPrior(shape, rate)
        posterior_precision = GammaPrior(shape, rate)
        return cls(prior_weights, posterior_weights, prior_precision,
                   posterior_precision)


    def __init__(self, prior_weights, posterior_weights, prior_precision,
                 posterior_precision):
        super().__init__()
        self.weights = BayesianParameter(prior_weights, posterior_weights)
        self.precision = BayesianParameter(prior_precision, posterior_precision)

    @property
    def dim(self):
        return self.weights.expected_value().shape[1]

    @property
    def variance(self):
        return 1. / self.precision.expected_value()

    def mean_field_factorization(self):
        return [[self.weights], [self.precision]]

    @staticmethod
    def sufficient_statistics(data):
        dim, dtype, device = data.shape[1], data.dtype, data.device
        return torch.cat([
            -.5 * torch.sum(data**2, dim=-1).reshape(-1, 1),
            data,
            -.5 * torch.ones(len(data), 1, dtype=dtype, device=device),
            .5 * dim * torch.ones(len(data), 1, dtype=dtype, device=device),
        ], dim=-1)

    def expected_log_likelihood(self, stats, regressors):
        X = stats[:, 1:-2]
        r_dim = regressors.shape[1]
        w_nparams = self.weights.expected_natural_parameters()
        quad_weights, weights = w_nparams[:int(r_dim**2)], \
                                w_nparams[int(r_dim**2):]
        weights = weights.reshape(r_dim, -1)
        quad = regressors[:, :, None] * regressors[:, None, :]
        quad = quad.reshape(len(stats), -1)
        mean = regressors @ weights

        prec, log_prec = self.precision.expected_natural_parameters()
        nparams = torch.cat([
            prec * torch.ones(len(X), 1, dtype=X.dtype, device=X.device),
            prec * mean,
            prec * (quad @ quad_weights).view(-1, 1),
            log_prec.repeat(len(X), 1)
        ], dim=1)

        # Cache necessary values to compute the natural gradients.
        self.cache['nparams'] = nparams[:, :-1]
        self.cache['quad'] = quad
        self.cache['regressors'] = regressors

        return torch.sum(stats * (nparams), dim=-1) \
                - .5 * X.shape[1] * math.log(math.pi)

    def accumulate(self, stats):
        X = stats[:, 1:-2]
        prec = self.precision.expected_value()
        delta = torch.sum(stats[:, :-1] * self.cache['nparams'] / prec, dim=-1)
        regressors = self.cache['regressors']
        quad = (regressors[:, :, None] * regressors[:, None, :]).view(len(X), -1)
        return {
            self.precision: torch.cat([
                delta.sum().view(1),
                stats[:, -1].sum().view(1)
            ]),
            self.weights: torch.cat([
                torch.sum(stats[:, -2, None] * quad, dim=0).view(-1),
                (regressors.t() @ X).view(-1)
            ]) * prec
        }


class LinearRegressionSet(BayesianModelSet):
    '''Set of Bayesian Linear Regression.

    '''

    @classmethod
    def create(cls, size, weights, variance, prior_strength=1.,
               weights_variance=.1, noise_std=.1):
        '''Create a Normal model.

        Args:
            weights (``torch.Tensor[size,dim]``): Prior weights of the model.
            variance (``torch.Tensor[1]``  or scalar):
                prior variance parameter.

        Returns:
            :any:`Normal`

        '''
        lregs = []
        for _ in range(size):
            lreg = LinearRegression.create(weights, variance, prior_strength,
                                           weights_variance, noise_std)
            lregs.append(lreg)
        return cls(lregs)


    def __init__(self, lregs):
        super().__init__()
        self.lregs = lregs
        for i, model in enumerate(lregs):
            self._register_submodel('lreg' + str(i), model)

    def __len__(self):
        return len(self.lregs)

    def __getitem__(self, key):
        return self.lregs[key]

    def mean_field_factorization(self):
        weights = [model.weights for model in self.lregs]
        precision = [model.precision for model in self.lregs]
        return [weights, precision]

    def sufficient_statistics(self, data):
        return self.lregs[0].sufficient_statistics(data)

    def expected_log_likelihood(self, stats, regressors):
        llhs = torch.cat([
            model.expected_log_likelihood(stats, regressors).view(-1, 1)
            for model in self.lregs
        ], dim=-1)
        return llhs

    def accumulate(self, stats, resps):
        acc_stats = {}
        for i, model in enumerate(self.lregs):
            m_acc_stats = model.accumulate(resps[:, i, None] * stats)
            acc_stats.update(m_acc_stats)
        return acc_stats


__all__ = [
    'LinearRegression',
    'LinearRegressionSet',
]
