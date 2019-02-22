'Bayesian Generalized Subspace Model.'

import copy
import math
import torch
from .basemodel import Model
from .parameters import BayesianParameter
from .parameters import ConjugateBayesianParameter
from ..dists import NormalDiagonalCovariance
from ..dists import JointNormalDiagonalCovariance


__all__ = ['GSM', 'SubspaceBayesianParameter']


########################################################################
# Affine Transform:  W^T h + b
# "W" and "b" have a Normal (with diagonal covariance matrix) prior.


# Parametererization of the Normal distribution. We use the log variance
# instead of the variance so we don't have any constraints during the 
# S.G.D. 
class _MeanLogDiagCov(torch.nn.Module):

    def __init__(self, mean, log_diag_cov):
        super().__init__()
        if mean.requires_grad:
            self.register_parameter('mean', torch.nn.Parameter(mean))
        else:
            self.register_buffer('mean', mean)
        
        if log_diag_cov.requires_grad:
            self.register_parameter('log_diag_cov', 
                                    torch.nn.Parameter(log_diag_cov))
        else:
            self.register_buffer('log_diag_cov', log_diag_cov)

    @property
    def diag_cov(self):
        return self.log_diag_cov.exp() 



# The Affine Transform is not a generative model it inherits from 
# "Model" as it has some Bayesian parameters.
class AffineTransform(Model):
    'Bayesian Affine Transformation: y = W^T x + b.'

    @classmethod
    def create(cls, in_dim, out_dim, prior_strength=1.):
        log_strength = math.log(prior_strength)
        # Bias prior/posterior.
        prior_bias = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(out_dim, requires_grad=False), 
                log_diag_cov=torch.zeros(out_dim, requires_grad=False) \
                             + log_strength,
            )
        )
        posterior_bias = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(out_dim, requires_grad=True), 
                log_diag_cov=torch.zeros(out_dim, requires_grad=True) \
                             + log_strength,
            )
        )

        # Weights prior/posterior.
        prior_weights = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(in_dim * out_dim, requires_grad=False), 
                log_diag_cov=torch.zeros(in_dim * out_dim, requires_grad=False) \
                             + log_strength,
            )
        )
        posterior_weights = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(in_dim * out_dim, requires_grad=True), 
                log_diag_cov=torch.zeros(in_dim * out_dim, requires_grad=True) \
                             + log_strength,
            )
        )
        return cls(prior_weights, posterior_weights, prior_bias, posterior_bias)

    def __init__(self, prior_weights, posterior_weights, prior_bias, 
                 posterior_bias):
        super().__init__()
        self.weights = ConjugateBayesianParameter(prior_weights, posterior_weights)
        self.bias = ConjugateBayesianParameter(prior_bias, posterior_bias)

        # Compute the input/output dimension from the priors.
        self._out_dim = self.bias.prior.dim
        self._in_dim = self.weights.prior.dim // self._out_dim

    @property
    def in_dim(self):
        return self._in_dim
    
    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, X, nsamples=1):
        s_b = self.bias.posterior.sample(nsamples)
        s_W = self.weights.posterior.sample(nsamples)
        s_W = s_W.reshape(-1, self.in_dim, self.out_dim)
        res = torch.matmul(X[None], s_W) + s_b[:, None, :]

        # For coherence with other components we reorganize the 3d 
        # tensor to have:
        #   - 1st dimension: number of input data points (i.e. len(X))
        #   - 2nd dimension: number samples to estimate the expectation
        #   - 3rd dimension: dimension of the latent space.
        return res.permute(1, 0, 2)

    ####################################################################
    # Model interface.
    ####################################################################

    def mean_field_factorization(self):
        return [[self.weights, self.bias]]

    def sufficient_statistics(self, data):
        return data

    def expected_log_likelihood(self, stats, labels=None, **kwargs):
        raise NotImplementedError

    def accumulate(self, stats):
        raise NotImplementedError


########################################################################
# Subspace parameter.


class SubspaceBayesianParameter(BayesianParameter):
    '''Specific class of (non-conjugate) Bayesian parameter for which 
    the prior/posterior live in a subspace.
    '''
    def __init__(self, parameter, prior):
        init_stats = torch.zeros_like(parameter.stats)
        super().__init__(init_stats, prior)
        self.likelihood_fn = parameter.prior.conjugate()
        self.pdfvec = None
    
    @property
    def sufficient_statistics_dim(self):
        return self.likelihood_fn.sufficient_statistics_dim

    def expected_value(self):
        self.likelihood_fn.parameters_from_pdfvector(self.pdfvec)

    def natural_form(self):
        return self.pdfvec

########################################################################
# GSM implementation.

# helper to iterate over all parameters handled by the subspace.
def _subspace_params(model):
    paramfilter = lambda param: isinstance(param, SubspaceBayesianParameter)
    for param in model.bayesian_parameters(paramfilter):
        yield param


# Error raised when attempting to create a GSM with a model having no
# Subspace parameters.
class NoSubspaceBayesianParameters(Exception): pass


# Parametererization of the Joint Normal distribution for the latent
# posterior. 
class _MeansLogDiagCovs(_MeanLogDiagCov):
    @property
    def means(self):
        return self.mean

    @property
    def diag_covs(self):
        return self.log_diag_cov.exp() 

# Parameterization of the Normal distribution for the which doesn't copy
# its parameters. Instead, it uses the parameters of a joint normal
# distribution.
class _NormalParamsView:
    def __init__(self, jointnormalparams, refidx):
        self.jointnormalparams = jointnormalparams
        self.refidx = refidx

    @property
    def means(self):
        return self.jointnormalparams.means[self.refidx]

    @property
    def diag_covs(self):
        return self.jointnormalparams.diag_covs[self.refidx]


def _svectors_from_rvectors(parameters, rvecs):
    'Map a set of real value vectors to the super-vector space.'
    retval = []
    idx = 0
    for param in parameters:
        pdf = param.conjugate()
        dim = pdf.conjugate_sufficient_statistics_dim
        stats = pdf.sufficient_statistics_from_rvectors(rvecs[:, idx:idx + dim])
        retval.append(stats)
        idx += dim
    return torch.cat(retval, dim=-1)

def _models_stats(models):
    return torch.cat([model.accumulated_statistics()[None] 
                      for model in models])

def _default_parameters_generator(model):
    return model.bayesian_parameters()

class GSM(Model):
    'Generalized Subspace Model.'

    @classmethod
    def create(cls, model, latent_dim, latent_prior, prior_strength=1.):
        '''Create a new GSM. 

        Args:
            model (`Model`): Model to integrate with the GSM. The model
                has to have at least one `SubspaceBayesianParameter`.
            latent_dim (int): Dimension of the latent space.
            latent_prior (``Model``): Prior distribution in the latent 
                space.
            prior_strength: (float): Strength of the prior over the 
                subspace parameters (weights and bias).
        
        Returns:
            a randomly initialized `GSM` object.

        '''
        svec_dim = sum([param.sufficient_statistics_dim 
                        for param in _subspace_params(model)])
        if svec_dim == 0:
            raise NoSubspaceBayesianParameters(
                'The model has to have at least one SubspaceBayesianParameter.')
        trans = AffineTransform.create(latent_dim, svec_dim, 
                                       prior_strength=prior_strength)
        return cls(model, trans, latent_prior)

    def __init__(self, model, affine_transform, latent_prior):
        super().__init__()
        self.model = model
        self.affine_transform = affine_transform
        self.latent_prior = latent_prior

    def new_models(self, nmodels):
        latent_posteriors = self.new_latent_posteriors(nmodels)
        jointnormalparams = latent_posteriors.params
        models = []
        for i in range(nmodels):
            new_model = copy.deepcopy(self.model)
            for param in _subspace_params(new_model):
                param.posterior = NormalDiagonalCovariance(jointnormalparams)
            models.append(new_model)
        return models

    def new_latent_posteriors(self, nposts):
        'Create a set of `nposts` latent posteriors.'
        # Check the type/device of the model.
        tensor = self.affine_transform.bias.prior.params.mean
        dtype, device = tensor.dtype, tensor.device

        init_means = torch.zeros(nposts, self.affine_transform.in_dim,
                                 dtype=dtype, device=device, requires_grad=True)
        init_log_diag_covs = torch.zeros(nposts, self.affine_transform.in_dim,
                                         dtype=dtype, device=device, 
                                         requires_grad=True)
        params = _MeansLogDiagCovs(mean=init_means, 
                                 log_diag_cov=init_log_diag_covs)
        return JointNormalDiagonalCovariance(params)

    def _xentropy(self, s_h, **kwargs):
        s_h = s_h.reshape(len(s_h), -1)
        stats = self.latent_prior.sufficient_statistics(s_h)
        llh = self.latent_prior.expected_log_likelihood(stats, **kwargs)
        llh = llh.reshape(-1, s_h.shape[0]).mean(dim=-1)
        return -llh, stats

    ####################################################################
    # Model interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_prior.mean_field_factorization()

    def sufficient_statistics(self, models):
        return models

    def expected_log_likelihood(self, models, latent_posts, latent_nsamples=1,
                                params_nsamples=1, **kwargs):
        nmodels = len(models)
        models_stats = _models_stats(models)
        nsamples = latent_nsamples * params_nsamples
        s_h = latent_posts.sample(latent_nsamples)

        # Local KL divergence posterior/prior
        # D(q || p) = H[q, p] - H[q]
        # where H[q, p] is the cross-entropy and H[q] is the entropy.
        xent, self.cache['lp_stats'] = self._xentropy(s_h, **kwargs)
        ent = -latent_posts(s_h).mean(dim=-1)
        local_kl_div = xent - ent

        # Compute the expected log-likelihood.
        s_h = s_h.reshape(-1, self.affine_transform.in_dim)
        rvecs = self.affine_transform(s_h, params_nsamples)
        rvecs = rvecs.reshape(-1, rvecs.shape[-1])
        svecs = _svectors_from_rvectors(models[0], rvecs)
        svecs = svecs.reshape(nmodels, nsamples, -1)
        llh = torch.sum(svecs * models_stats[:, None, :], dim=(1,2))

        return llh - local_kl_div

    def accumulate(self, stats):
        return self.latent_prior.accumulate(self.cache['lp_stats'])
