'Bayesian Generalized Subspace Model.'

import copy
from dataclasses import dataclass
import math
from operator import mul
import torch
from .basemodel import Model
from .parameters import BayesianParameter
from .parameters import ConjugateBayesianParameter
from ..dists import NormalDiagonalCovariance


__all__ = ['GSM', 'SubspaceBayesianParameter']


########################################################################
# Affine Transform:  W^T h + b
# "W" and "b" have a Normal (with diagonal covariance matrix) prior.


# Parametererization of the Normal distribution. We use the log variance
# instead of the variance so we don't have any constraints during the 
# S.G.D. 
@dataclass(init=False, unsafe_hash=True)
class _MeanLogDiagCov(torch.nn.Module):
    mean: torch.Tensor
    log_diag_cov: torch.Tensor 
    
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

    @classmethod
    def from_parameter(cls, parameter, prior):
        'Build a SubspaceBayesianParameter from an existing BayesianParameter.'
        init_stats = parameter.stats.clone().detach()
        lhf = parameter.prior.conjugate()
        return cls(init_stats, prior, likelihood_fn=lhf)

    def __init__(self, init_stats, prior, posterior=None, likelihood_fn=None,
                 pdfvec=None):
        super().__init__(init_stats, prior, posterior, likelihood_fn)
        self.pdfvec = pdfvec

    def value(self):
        return self.likelihood_fn.parameters_from_pdfvector(self.pdfvec)

    def natural_form(self):
        return self.pdfvec
    
    def kl_div_posterior_prior(self):
        # Returns 0 as the KL divergence is computed with the GSM model 
        # instance.
        dtype, device = self.pdfvec.dtype, self.pdfvec.device
        return torch.tensor(0., dtype=dtype, device=device, requires_grad=False)
    
    #def __len__(self):
    #    if len(self.stats.shape) <= 1:
    #        return 1
    #    return self.stats.shape[0]
    
    def __getitem__(self, key):
        return SubspaceBayesianParameter(self.stats[key], self.prior, 
                                         self.posterior, 
                                         self.likelihood_fn,
                                         self.pdfvec[key])

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


def _pdfvecs_from_rvectors(parameters, rvecs):
    'Map a set of real value vectors to the pdf vectors.'
    retval = []
    idx = 0
    for param in parameters:
        lhf = param.likelihood_fn
        npdfs = len(param)
        dim = lhf.sufficient_statistics_dim(zero_stats=False) 
        totdim = dim * npdfs
        rvec = rvecs[:, idx: idx + totdim].reshape(-1, dim)
        pdfvec = lhf.pdfvectors_from_rvectors(rvec)
        pdfvec = pdfvec.reshape(len(rvecs), npdfs, -1).mean(dim=0)
        retval.append(pdfvec.reshape(param.stats.shape))
        idx += totdim
    return retval


class GSM(Model):
    'Generalized Subspace Model.'

    @classmethod
    def create(cls, model, latent_dim, latent_prior, prior_strength=1.,
               latent_nsamples=1, params_nsamples=1):
        '''Create a new GSM. 

        Args:
            model (`Model`): Model to integrate with the GSM. The model
                has to have at least one `SubspaceBayesianParameter`.
            latent_dim (int): Dimension of the latent space.
            latent_prior (``Model``): Prior distribution in the latent 
                space.
            prior_strength: (float): Strength of the prior over the 
                subspace parameters (weights and bias).
            latent_nsamples (int): Number of samples to draw from the 
                latent posterior to estimate the expected value of the 
                parameters of the model.
            params_nsamples (int): Number of samples to draw from the 
                subspace parameters' posterior to estimate the expected
                value of the parameters of the model.
        
        Returns:
            a randomly initialized `GSM` object.

        '''
        def get_dim(param):
             lhf = param.likelihood_fn
             return lhf.sufficient_statistics_dim(zero_stats=False) * len(param)
        svec_dim = sum([get_dim(param) for param in _subspace_params(model)])
        if svec_dim == 0:
            raise NoSubspaceBayesianParameters(
                'The model has to have at least one SubspaceBayesianParameter.')
        trans = AffineTransform.create(latent_dim, svec_dim, 
                                       prior_strength=prior_strength)
        return cls(model, trans, latent_prior, latent_nsamples, params_nsamples)

    def __init__(self, model, affine_transform, latent_prior, latent_nsamples=1,
                 params_nsamples=1):
        super().__init__()
        self.model = model
        self.affine_transform = affine_transform
        self.latent_prior = latent_prior
    
    def update_models(self, models, latent_posteriors, latent_nsamples=1, 
                      params_nsamples=1):
        samples = latent_posteriors.sample(latent_nsamples)
        rvecs = self._rvecs_from_samples(samples, params_nsamples)
        for i, model, model_rvecs in zip(range(len(models)), models, rvecs):
            pdfvecs = _pdfvecs_from_rvectors(_subspace_params(model), 
                                             model_rvecs)
            for param, pdfvec in zip(_subspace_params(model), pdfvecs):
                param.posterior = latent_posteriors.view(i)
                param.pdfvec = pdfvec

    def new_models(self, nmodels, latent_nsamples=1, params_nsamples=1):
        latent_posteriors = self.new_latent_posteriors(nmodels)
        jointnormalparams = latent_posteriors.params
        samples = latent_posteriors.sample(latent_nsamples)
        rvecs = self._rvecs_from_samples(samples, params_nsamples)
        models = []
        for i, model_rvecs in zip(range(nmodels), rvecs):
            new_model = copy.deepcopy(self.model)
            pdfvecs = _pdfvecs_from_rvectors(_subspace_params(new_model), 
                                             model_rvecs)
            for param, pdfvec in zip(_subspace_params(new_model), pdfvecs):
                param.posterior = latent_posteriors.view(i)
                param.pdfvec = pdfvec
            models.append(new_model)
        return models, latent_posteriors

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
        params = _MeanLogDiagCov(mean=init_means, 
                                 log_diag_cov=init_log_diag_covs)
        return NormalDiagonalCovariance(params)

    def _xentropy(self, s_h, **kwargs):
        shape = s_h.shape
        s_h = s_h.reshape(-1, s_h.shape[-1])
        stats = self.latent_prior.sufficient_statistics(s_h)
        llh = self.latent_prior.expected_log_likelihood(stats, **kwargs)
        llh = llh.reshape(shape[0], shape[1]).mean(dim=-1)
        return -llh, stats.reshape(shape[0], shape[1], stats.shape[-1]).mean(dim=1)
    
    def _entropy(self, s_h, latent_posteriors):
        shape = s_h.shape
        #s_h = s_h.reshape(-1, s_h.shape[-1])
        dim = shape[-1]
        stats = torch.cat([s_h, s_h**2], dim=-1)
        nparams = latent_posteriors.natural_parameters()
        logdets = latent_posteriors.params.diag_cov.log().sum(dim=-1, 
                                                              keepdim=True)
        log_basemeasure = -.5 * (logdets + dim * math.log(2 * math.pi))
        llh = (nparams[:, None, :] * stats).sum(dim=-1) + log_basemeasure
        return -llh.mean(dim=-1)

    def _rvecs_from_samples(self, samples, params_nsamples):
        shape = samples.shape
        samples = samples.reshape(-1, shape[-1])
        rvecs = self.affine_transform(samples, params_nsamples)
        return rvecs.reshape(shape[0], -1, rvecs.shape[-1])

    def _models_log_likelihood_from_rvecs(self, models, rvecs):
        llhs = []
        for model, model_rvecs in zip(models, rvecs):
            pdfvecs = _pdfvecs_from_rvectors(_subspace_params(model), 
                                             model_rvecs)
            model_llh = torch.tensor(0., dtype=rvecs.dtype, device=rvecs.device)
            for param, pdfvec in zip(_subspace_params(model), pdfvecs):
                model_llh += (param.stats * pdfvec).sum()
            llhs.append(model_llh.view(1))
        return torch.cat(llhs)

    ####################################################################
    # Model interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.latent_prior.mean_field_factorization()

    def sufficient_statistics(self, models):
        return models

    def expected_log_likelihood(self, models, latent_posts, latent_nsamples=1,
                                params_nsamples=1, **kwargs):
        nsamples = latent_nsamples * params_nsamples
        s_h = latent_posts.sample(latent_nsamples)

        # Local KL divergence posterior/prior
        # D(q || p) = H[q, p] - H[q]
        # where H[q, p] is the cross-entropy and H[q] is the entropy.
        xent, self.cache['lp_stats'] = self._xentropy(s_h, **kwargs)
        ent = self._entropy(s_h, latent_posts)
        local_kl_div = xent - ent

        # Compute the expected log-likelihood.
        rvecs = self._rvecs_from_samples(s_h, params_nsamples)
        llhs = self._models_log_likelihood_from_rvecs(models, rvecs)
        return llhs - local_kl_div

    def accumulate(self, stats):
        return self.latent_prior.accumulate(self.cache['lp_stats'])
