'Bayesian Generalized Subspace Model.'

import copy
from dataclasses import dataclass
import math
from operator import mul
import torch
from .basemodel import Model
from .modelset import ModelSet
from .parameters import BayesianParameter
from .parameters import ConjugateBayesianParameter
from ..dists import NormalDiagonalCovariance
from ..dists import NormalFullCovariance
from .normal import UnknownCovarianceType


__all__ = ['GSM', 'GSMSet', 'SubspaceBayesianParameter']


########################################################################
# Affine Transform:  W^T h + b
# "W" and "b" have a Normal (with diagonal covariance matrix) prior.


# Parametererization of the Normal distribution with diagonal
# covariance matrix to be optimized with S.G.D.
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
        return 1e-4 + self.log_diag_cov.exp()


# Parametererization of the Normal distribution with full
# covariance matrix to be optimized with S.G.D.
@dataclass(init=False, unsafe_hash=True)
class _MeanLogDiagLL(torch.nn.Module):
    mean: torch.Tensor
    log_diag_L: torch.Tensor
    L_offdiag: torch.Tensor

    def __init__(self, mean, log_diag_L, L_offdiag):
        super().__init__()
        if mean.requires_grad:
            self.register_parameter('mean', torch.nn.Parameter(mean))
        else:
            self.register_buffer('mean', mean)

        if log_diag_L.requires_grad:
            self.register_parameter('log_diag_L',
                                    torch.nn.Parameter(log_diag_L))
        else:
            self.register_buffer('log_diag_L', log_diag_L)

        if L_offdiag.requires_grad:
            self.register_parameter('L_offdiag',
                                    torch.nn.Parameter(L_offdiag))
        else:
            self.register_buffer('L_offdiag', L_offdiag)

    @property
    def cov(self):
        diag_L = 1e-3 + self.log_diag_L.exp()
        L_offdiag = self.L_offdiag
        size = len(self.mean.shape)
        ncomps = len(self.mean) if size > 1 else 1
        dim = self.mean.shape[-1]
        dtype = diag_L.dtype
        device = diag_L.device

        tril_indices = torch.ones(dim, dim, dtype=torch.long,
                                  device=device).tril(diagonal=-1).nonzero()
        L = torch.zeros(ncomps, dim, dim, dtype=dtype, device=device)
        L[:, range(dim), range(dim)] = diag_L
        L[:, tril_indices[:, 0], tril_indices[:, 1]] = L_offdiag
        cov = torch.matmul(L, L.permute(0, 2, 1))
        return cov


# The Affine/Linear Transform are not generative models but they
# inherit from "Model" as it has some Bayesian parameters.
class AffineTransform(Model):
    'Bayesian Affine Transformation: y = W^T x + b.'

    @classmethod
    def create(cls, in_dim, out_dim, prior_strength=1.):
        log_strength = -math.log(prior_strength)
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
                log_diag_cov=torch.zeros(out_dim, requires_grad=True),
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
                log_diag_cov=torch.zeros(in_dim * out_dim, requires_grad=True),
            )
        )
        return cls(prior_weights, posterior_weights, prior_bias, posterior_bias)

    def __init__(self, prior_weights, posterior_weights, prior_bias,
                 posterior_bias):
        super().__init__()
        self.weights = BayesianParameter(prior_weights, posterior_weights)
        self.bias = BayesianParameter(prior_bias, posterior_bias)

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
        # tensor to have:pplication to Other Languages
        #   - 1st dimension: number of input data points (i.e. len(X))
        #   - 2nd dimension: number samples to estimate the expectation
        #   - 3rd dimension: dimension of the latent space.
        return res.permute(1, 0, 2)

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        return [[self.weights, self.bias]]

    def sufficient_statistics(self, data):
        return data

    def expected_log_likelihood(self, stats, **kwargs):
        raise NotImplementedError

    def accumulate(self, stats):
        raise NotImplementedError


class LinearTransform(Model):
    'Bayesian Linear Transformation: y = W^T x.'

    @classmethod
    def create(cls, in_dim, out_dim, prior_strength=1.):
        log_strength = -math.log(prior_strength)
        # Weights prior/posterior.
        prior_weights = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(in_dim * out_dim, requires_grad=False),
                log_diag_cov=torch.zeros(in_dim * out_dim,
                                         requires_grad=False) + log_strength,
            )
        )
        posterior_weights = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(in_dim * out_dim, requires_grad=True),
                log_diag_cov=torch.zeros(in_dim * out_dim, requires_grad=True),
            )
        )
        return cls(prior_weights, posterior_weights, out_dim)

    def __init__(self, prior_weights, posterior_weights, out_dim):
        super().__init__()
        self.weights = BayesianParameter(prior_weights, posterior_weights)

        # Compute the input/output dimension from the priors.
        self._out_dim = out_dim
        self._in_dim = self.weights.prior.dim // self._out_dim

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, X, nsamples=1):
        s_W = self.weights.posterior.sample(nsamples)
        s_W = s_W.reshape(-1, self.in_dim, self.out_dim)
        res = torch.matmul(X[None], s_W)

        # For coherence with other components we reorganize the 3d
        # tensor to have:
        #   - 1st dimension: number of input data points (i.e. len(X))
        #   - 2nd dimension: number samples to estimate the expectation
        #   - 3rd dimension: dimension of the latent space.
        return res.permute(1, 0, 2)

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        return [[self.weights]]

    def sufficient_statistics(self, data):
        return data

    def expected_log_likelihood(self, stats, **kwargs):
        raise NotImplementedError

    def accumulate(self, stats):
        raise NotImplementedError



########################################################################
# Subspace parameter.


class SubspaceBayesianParameter(ConjugateBayesianParameter):
    '''Specific class of (non-conjugate) Bayesian parameter for which
    the prior/posterior live in a subspace.
    '''

    @classmethod
    def from_parameter(cls, parameter, prior):
        'Build a SubspaceBayesianParameter from an existing BayesianParameter.'
        init_stats = parameter.stats.clone().detach()
        lhf = parameter.prior.conjugate()
        return cls(prior, init_stats=init_stats, likelihood_fn=lhf)

    def __init__(self, prior, posterior=None, init_stats=None,
                 likelihood_fn=None, pdfvec=None):
        super().__init__(prior, posterior, init_stats, likelihood_fn)
        pdfvec = pdfvec if pdfvec is not None else torch.zeros_like(self.stats)
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

    def __getitem__(self, key):
        return SubspaceBayesianParameterView(key, self)


class SubspaceBayesianParameterView(ConjugateBayesianParameter):

    def __init__(self, key, param):
        ConjugateBayesianParameter.__init__(self, param.prior, param.posterior,
                                            param.stats, param.likelihood_fn)
        del self.stats
        self.key = key
        self.param = param

    @property
    def stats(self):
        return self.param.stats[self.key]

    @stats.setter
    def stats(self, value):
        self.param.stats[self.key] = value

    @property
    def pdfvec(self):
        return self.param.pdfvec[self.key]

    @pdfvec.setter
    def pdfvec(self, value):
        self.param.pdfvec[self.key] = value

    def value(self):
        return self.likelihood_fn.parameters_from_pdfvector(self.pdfvec)

    def natural_form(self):
        self.pdfvec

    def kl_div_posterior_prior(self):
        return self.param.kl_div_posterior_prior()

    def __getitem__(self, key):
        return SubspaceBayesianParameterView(key, self.param)


########################################################################
# Helpers for work with the super-vector space, the "real" super-vector,
# and the latent space.

# Iterate over all parameters handled by the GSM.
def _subspace_params(model):
    sbp_classes = (SubspaceBayesianParameter, SubspaceBayesianParameterView)
    pfilter = lambda param: isinstance(param, sbp_classes)
    for param in model.bayesian_parameters(paramfilter=pfilter):
        yield param

# Dimension of the super-vector for the current model.
def _svec_dim(model):
    def get_dim(param):
         lhf = param.likelihood_fn
         s_stats_dim = lhf.sufficient_statistics_dim(zero_stats=False)
         return s_stats_dim * len(param)
    return sum([get_dim(param) for param in _subspace_params(model)])

# Iterate over the pdfvectors corresponding to the given real vectors.
def _pdfvecs(params, rvecs):
    idx = 0
    for param in params:
        lhf = param.likelihood_fn
        dim = lhf.sufficient_statistics_dim(zero_stats=False)
        npdfs = len(param)
        totdim = dim * npdfs
        param_rvecs = rvecs[:, idx: idx + totdim].reshape(-1, dim)
        pdfvec = lhf.pdfvectors_from_rvectors(param_rvecs)
        totdim_with_zerostats = npdfs * pdfvec.shape[-1]
        idx += totdim
        yield pdfvec.reshape(-1, totdim_with_zerostats)

# Update expected value of the subspace paramters given a set of
# pdf-vectors.
def _update_params(params, pdfvecs):
    idx = 0
    for param in params:
        totdim = param.stats.numel()
        shape = param.stats.shape
        param.pdfvec = pdfvecs[idx: idx + totdim].reshape(shape)
        idx += totdim

# Update the models given a set of pdf-vectors.
def _update_models(models, pdfvecs):
    for model, model_pdfvecs in zip(models, pdfvecs):
        params = _subspace_params(model)
        _update_params(params, model_pdfvecs)

# Approximate cross-entropy given samples and a model.
def _xentropy(s_h, model, **kwargs):
    shape = s_h.shape
    s_h = s_h.reshape(-1, s_h.shape[-1])
    stats = model.sufficient_statistics(s_h)
    stats = stats.reshape(shape[0], shape[1], -1).mean(dim=1)
    llh = model.expected_log_likelihood(stats, **kwargs)
    return -llh.reshape(-1), stats.detach()

# Approximate entropy given samples and a (set of) distributions.
def _entropy(s_h, dists):
    length, nsamples, dim = s_h.shape
    s_h = s_h.reshape(-1, s_h.shape[-1])
    stats = dists.sufficient_statistics(s_h)
    stats = stats.reshape(length, nsamples, -1).mean(dim=1)
    return - dists(stats, pdfwise=True)

# real super-vectors from samples.
def _rvecs_from_samples(samples, transform, params_nsamples):
    shape = samples.shape
    samples = samples.reshape(-1, shape[-1])
    rvecs = transform(samples, params_nsamples)
    return rvecs.reshape(shape[0], -1, rvecs.shape[-1])

# Return the pdf-vectors NxSxQ corresponding to the real
# vectors NxSxD.
def _pdfvecs_from_rvecs(rvecs, model):
    shape = rvecs.shape
    rvecs = rvecs.reshape(-1, rvecs.shape[-1])
    params = _subspace_params(model)
    list_pdfvecs = [pdfvecs for pdfvecs in _pdfvecs(params, rvecs)]
    pdfvecs = torch.cat(list_pdfvecs, dim=-1)
    return pdfvecs.reshape(shape[0], -1, pdfvecs.shape[-1])


########################################################################
# GSM implementation.

# Error raised when attempting to create a GSM with a model having no
# Subspace parameters.
class NoSubspaceBayesianParameters(Exception): pass

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
        svec_dim = _svec_dim(model)
        if svec_dim == 0:
            raise NoSubspaceBayesianParameters(
                'The model has to have at least one SubspaceBayesianParameter.')
        trans = AffineTransform.create(latent_dim, svec_dim,
                                       prior_strength=prior_strength)
        return cls(model, trans, latent_prior)

    def __init__(self, model, transform, latent_prior):
        super().__init__()
        self.model = model
        self.transform = transform
        self.latent_prior = latent_prior

    def new_models(self, nmodels, cov_type='diagonal', latent_nsamples=1,
                   params_nsamples=1):
        latent_posteriors = self.new_latent_posteriors(nmodels, cov_type)
        models = [copy.deepcopy(self.model) for _ in range(nmodels)]

        # Connect the models and the latent posteriors.
        for model in models:
            params = _subspace_params(model)
            for param in params:
                param.posterior = latent_posteriors
                param.pdfvecs = torch.zeros_like(param.stats)

        # Initialize the newly created models.
        samples = latent_posteriors.sample(latent_nsamples)
        rvecs = _rvecs_from_samples(samples, self.transform, params_nsamples)
        pdfvecs = _pdfvecs_from_rvecs(rvecs, self.model).mean(dim=1)
        _update_models(models, pdfvecs)

        return models, latent_posteriors

    def new_latent_posteriors(self, nposts, cov_type='diagonal'):
        'Create a set of `nposts` latent posteriors.'
        if cov_type not in ['full', 'diagonal']:
            raise UnknownCovarianceType('f{cov_type}')
        tensor = self.transform.bias.prior.params.mean
        dim = self.transform.in_dim
        dtype, device = tensor.dtype, tensor.device
        tensorconf = {'dtype': dtype, 'device': device, 'requires_grad':True}
        init_means = torch.zeros(nposts, dim, **tensorconf)
        init_log_diag_covs = torch.zeros(nposts, dim, **tensorconf) \
                             - math.log(dim)

        if cov_type == 'full':
            init_corrs = torch.zeros(nposts, int( .5 * dim * (dim - 1)),
                                     **tensorconf)
            params = _MeanLogDiagLL(init_means, init_log_diag_covs,
                                         init_corrs)
            return NormalFullCovariance(params)

        # Diagonal covariance case.
        params = _MeanLogDiagCov(init_means, init_log_diag_covs)
        return NormalDiagonalCovariance(params)

    def expected_pdfvecs(self, latent_posteriors, latent_nsamples=1,
                         params_nsamples=1):
        s_h = latent_posteriors.sample(latent_nsamples)
        rvecs = _rvecs_from_samples(s_h, self.transform, params_nsamples)
        return _pdfvecs_from_rvecs(rvecs, self.model).mean(dim=1)

    def update_models(self, models, pdfvecs):
        return _update_models(models, pdfvecs)

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        return [*self.latent_prior.mean_field_factorization(),
                *self.transform.mean_field_factorization()]

    def sufficient_statistics(self, models):
        # We keep a pointer to the model object to update the posterior
        # of their parameters.
        self.cache['models'] = models

        stats = []
        for model in models:
            params = _subspace_params(model)
            stats.append(torch.cat([param.stats.view(1, -1)
                                    for param in params], dim=-1))
        return torch.cat(stats, dim=0)

    def expected_log_likelihood(self, stats, latent_posts, latent_nsamples=1,
                                params_nsamples=1, update_models=True,
                                **kwargs):
        nsamples = latent_nsamples * params_nsamples
        s_h = latent_posts.sample(latent_nsamples)

        # Local KL divergence posterior/prior
        # D(q || p) = H[q, p] - H[q]
        # where H[q, p] is the cross-entropy and H[q] is the entropy.
        xent, self.cache['lp_stats'] = _xentropy(s_h, self.latent_prior,
                                                 **kwargs)
        ent = _entropy(s_h, latent_posts)
        local_kl_div = xent - ent

        # Compute the expected log-likelihood.
        rvecs = _rvecs_from_samples(s_h, self.transform, params_nsamples)
        pdfvecs = _pdfvecs_from_rvecs(rvecs, self.model).mean(dim=1)
        llh = (pdfvecs * stats).sum(dim=-1)

        if update_models: _update_models(self.cache['models'], pdfvecs)

        return llh - local_kl_div

    def accumulate(self, stats):
        return self.latent_prior.accumulate(self.cache['lp_stats'])


class GSMSetMeansParameter(BayesianParameter):

    def __init__(self, prior, posterior):
        super().__init__(prior, posterior)
        tensor = posterior.sample(1)
        dtype, device = tensor.dtype, tensor.device
        val = torch.tensor(0., dtype=dtype, device=device)
        self.register_buffer('kl_div', val)

    def kl_div_posterior_prior(self):
       return self.kl_div


class GSMSet(ModelSet):
    '''Model set that share a  Generalized Subspace Model and has a
    model specific subspace. It is analogous to a PLDA model in the
    parameter space of the standard Model.

    '''
    @classmethod
    def create(cls, model, size, latent_dim, dlatent_dim, latent_prior,
               dlatent_prior, prior_strength=1.):
        '''Create a new GSM.

        Args:
            model (`Model`): Model to integrate with the GSM. The model
                has to have at least one `SubspaceBayesianParameter`.
            size (int): Size of the set.
            latent_dim (int): Dimension of the latent space.
            mean_latent_dim (int): Dimenion of the discriminant latent
                space.
            latent_prior (``Model``): Prior distribution in the latent
                space.
            dlatent_prior (``Model``): Prior distribution in the
                discriminant latent space.
            prior_strength: (float): Strength of the prior over the
                subspace parameters (weights and bias).

        Returns:
            a randomly initialized `GSM` object.

        '''
        svec_dim = _svec_dim(model)
        if svec_dim == 0:
            raise NoSubspaceBayesianParameters(
                'The model has to have at least one SubspaceBayesianParameter.')
        trans = AffineTransform.create(latent_dim, svec_dim,
                                       prior_strength=prior_strength)
        disc_trans = LinearTransform.create(dlatent_dim, svec_dim,
                                            prior_strength=prior_strength)

        # Create the class mean posteriors.
        dim = disc_trans.in_dim
        tensor = disc_trans.weights.prior.params.mean
        dtype, device = tensor.dtype, tensor.device
        tensorconf = {'dtype': dtype, 'device': device, 'requires_grad':True}
        means = torch.zeros(size, dim, **tensorconf)
        log_diag_covs = torch.zeros(size, dim, **tensorconf) - math.log(dim)
        params = _MeanLogDiagCov(means, log_diag_covs)
        means = GSMSetMeansParameter(dlatent_prior,
                                     NormalDiagonalCovariance(params))

        return cls(model, trans, disc_trans, means, latent_prior)

    def __init__(self, model, transform, disc_transform, means, latent_prior):
        super().__init__()
        self.model = model
        self.transform = transform
        self.means = means
        self.disc_transform = disc_transform
        self.latent_prior = latent_prior

    def new_models(self, nmodels, cov_type='diagonal', latent_nsamples=1,
                   params_nsamples=1):
        return GSM.new_models(self, nmodels, cov_type, latent_nsamples,
                              params_nsamples)

    def new_latent_posteriors(self, nposts, cov_type='diagonal'):
        return GSM.new_latent_posteriors(self, nposts, cov_type)

    def expected_pdfvecs(self, latent_posteriors, latent_nsamples=1,
                         params_nsamples=1):
        return GSM.expected_pdfvecs(self, latent_posteriors, latent_nsamples,
                                    params_nsamples)

    def update_models(self, models, pdfvecs):
        return _update_models(models, pdfvecs)

    ####################################################################
    # Model interface.

    def mean_field_factorization(self):
        return [*self.latent_prior.mean_field_factorization(),
                [self.means],
                *self.means.prior.mean_field_factorization(),
                *self.transform.mean_field_factorization(),
                *self.disc_transform.mean_field_factorization()]

    def sufficient_statistics(self, models):
        # We keep a pointer to the model object to update the posterior
        # of their parameters.
        self.cache['models'] = models

        stats = []
        for model in models:
            params = _subspace_params(model)
            stats.append(torch.cat([param.stats.view(1, -1)
                                    for param in params], dim=-1))
        return torch.cat(stats, dim=0)

    def expected_log_likelihood(self, stats, latent_posts, latent_nsamples=1,
                                params_nsamples=1, update_models=True,
                                **kwargs):
        nsamples = latent_nsamples * params_nsamples
        s_h = latent_posts.sample(latent_nsamples)
        s_means = self.means.posterior.sample(latent_nsamples)

        # Local KL divergence posterior/prior
        # D(q || p) = H[q, p] - H[q]
        # where H[q, p] is the cross-entropy and H[q] is the entropy.
        # 1) within class subspace kl div..
        xent, self.cache['lp_stats'] = _xentropy(s_h, self.latent_prior,
                                                 **kwargs)
        ent = _entropy(s_h, latent_posts)
        local_kl_div = xent - ent

        # 2) across class subspace kl div.
        dxent, self.cache['dlp_stats'] = _xentropy(s_means, self.means.prior,
                                                   **kwargs)
        dent = _entropy(s_means, self.means.posterior)
        dlocal_kl_div = (dxent - dent).sum()
        self.means.kl_div = dlocal_kl_div

        # Compute the expected log-likelihood.
        rvecs = _rvecs_from_samples(s_h, self.transform, params_nsamples)
        means_rvecs = _rvecs_from_samples(s_means, self.disc_transform,
                                          params_nsamples)
        frvecs = rvecs[None, :, :] + means_rvecs[:, None, :, :]
        frvecs = frvecs.reshape(-1, *frvecs.shape[2:])
        pdfvecs = _pdfvecs_from_rvecs(frvecs, self.model).mean(dim=1)
        pdfvecs = pdfvecs.reshape(len(self), len(latent_posts), -1)
        llh = (pdfvecs * stats).sum(dim=-1)

        if update_models: self.cache['pdfvecs'] = pdfvecs.detach()

        return (llh - local_kl_div).t()

    def accumulate(self, stats, resps):
        try:
            pdfvecs = self.cache['pdfvecs']
            pdfvecs = (resps.t()[:, :, None] * pdfvecs).sum(dim=0)
            _update_models(self.cache['models'], pdfvecs)
        except KeyError:
            pass
        return {
            **self.means.prior.accumulate(self.cache['dlp_stats']),
            **self.latent_prior.accumulate(self.cache['lp_stats'])
        }

    ####################################################################
    # ModelSet interface.

    def __getitem__(self, key):
        pass

    def __len__(self):
        return len(self.means.posterior)


