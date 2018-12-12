import abc
import math
import torch

from .bayesmodel import BayesianModel
from .parameters import BayesianParameter, ConstantParameter
from ..priors import NormalDiagonalCovariancePrior
from ..priors import GammaPrior
from ..priors import JointGammaPrior
from ..priors import HierarchicalMatrixNormalPrior
from ..priors import MatrixNormalPrior
from ..priors.wishart import _logdet
from ..utils import make_symposdef


def kl_div_std_norm(means, cov):
    '''KL divergence between a set of Normal distributions with a
    shared covariance matrix and a standard Normal N(0, I).
    Args:
        means (``torch.Tensor[N, dim]``): Means of the
            Normal distributions where N is  the number of frames and
            dim is the dimension of the random variable.
        cov (``torch.Tensor[s_dim, s_dim]``): Shared covariance matrix.
    Returns:
        ``torch.Tensor[N]``: Per-distribution KL-divergence.
    '''
    dim = means.size(1)
    logdet = _logdet(cov)
    return .5 * (-dim - logdet + torch.trace(cov) + \
        torch.sum(means ** 2, dim=1))


def batch_inverse(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv


def _trace_array(matrices):
    'Compute trace of a set of matrices.'
    dim = len(matrices[0])
    idxs = tuple(range(dim))
    return matrices[:, idxs, idxs].sum(dim=-1)


class GeneralizedSubspaceModel(BayesianModel):
    '''Bayesian Generalized Subspace Model.

    Attributes:
        weights: weights matrix parameter.
        precision: precision parameter.

    '''

    @staticmethod
    def create(llh_func, mean_subspace, global_mean, mean_prec, mean_scale,
               noise_std=0., prior_strength=1., hyper_prior_strength=1.,
               hessian_type='full'):
        '''Create a Bayesian Generalized Subspace Model.

        Args:
            llh_func: (:any:`LikelihoodFunction`): The type of model
                of the subpsace.
            mean_subspace (``torch.Tensor[K,D]``): Mean of the prior
                over the bases of the subspace. K is the dimension of
                the subspace and D the dimension of the parameter
                space.
            global_mean (``torch.Tensor[D]``): Mean  of the prior
                over the global mean in the parameter space.
            mean_prec (``torch.Tensor[D]``): Mean of the prior over
                the precision.
            mean_scale (``torch.Tensor[K]``): Mean of the scale of the
                covariance matrix of the subspace prior.
            noise_std (float): Standard deviation of the noise for the
                random initialization of the subspace.
            prior_strength (float): Strength of the prior over the
                bases of the subspace and the global mean.
            hessian_type (string): Type of approximation. Possible
                choices are "full" (no approximation), "diagonal"
                or "scalar".

        '''
        if hessian_type == 'full':
            cls = GeneralizedSubspaceModelFull
        elif hessian_type == 'diagonal':
            cls = GeneralizedSubspaceModelDiagonal
        elif hessian_type == 'scalar':
            cls = GeneralizedSubspaceModelScalar
        else:
            raise ValueError(f'Unknown hessian type: "{hessian_type}"')

        dim_s = len(mean_subspace)
        dim_o = len(global_mean)
        dtype, device = mean_subspace.dtype, mean_subspace.device
        with torch.no_grad():
            # Scale prior/posterior.
            pad = torch.ones(dim_s, dtype=dtype, device=device)
            shapes = hyper_prior_strength * pad
            rates = (hyper_prior_strength / mean_scale) * pad
            scale_prior = JointGammaPrior(shapes, rates)
            scale_posterior = JointGammaPrior(shapes, rates)

            # Subspace prior/posterior.
            U = torch.diag(1/scale_posterior.expected_value())
            r_M = mean_subspace + noise_std * torch.randn(dim_s,
                                                          dim_o,
                                                          dtype=dtype,
                                                          device=device)
            subspace_prior = HierarchicalMatrixNormalPrior(mean_subspace,
                                                           scale_posterior)
            subspace_posterior = MatrixNormalPrior(r_M, subspace_prior.cov)

            # Global mean prior/posterior.
            dcov = torch.ones(len(global_mean), dtype=dtype, device=device)
            dcov /= prior_strength
            mean_prior = NormalDiagonalCovariancePrior(global_mean, dcov)
            mean_posterior = NormalDiagonalCovariancePrior(global_mean + torch.randn(*global_mean.shape), dcov)

            # Precision prior/posterior.
            shape = prior_strength * torch.ones(len(global_mean),
                                                dtype=dtype, device=device)
            rate =  prior_strength / mean_prec
            prec_prior = JointGammaPrior(shape, rate)
            prec_posterior = JointGammaPrior(shape, rate)

        return cls(llh_func, subspace_prior, subspace_posterior,
                   mean_prior, mean_posterior, prec_prior, prec_posterior,
                   scale_prior, scale_posterior)


    def __init__(self, llh_func, subspace_prior, subspace_posterior,
                 mean_prior, mean_posterior, prec_prior, prec_posterior,
                 scale_prior, scale_posterior):
        super().__init__()
        self.subspace = BayesianParameter(subspace_prior, subspace_posterior)
        self.mean = BayesianParameter(mean_prior, mean_posterior)
        self.precision = BayesianParameter(prec_prior, prec_posterior)
        self.scale = BayesianParameter(scale_prior, scale_posterior)
        self.llh_func = llh_func
        self.scale.register_callback(self._update_subspace_prior)

    def _update_subspace_prior(self):
        M, _ = self.subspace.prior.to_std_parameters()
        new_prior = HierarchicalMatrixNormalPrior(M, self.scale.posterior)
        self.subspace.prior = new_prior

    def _extract_moments(self):
        m, mm = self.mean.posterior.moments()
        W, WW = self.subspace.posterior.moments()
        _, U = self.subspace.posterior.to_std_parameters()
        dim = len(m)
        nparams = self.precision.expected_natural_parameters()
        prec, log_prec = nparams[:dim], nparams[dim:]
        WLW = prec.sum() * U + W @ torch.diag(prec) @ W.t()
        return prec, log_prec, m, mm, W, WW, WLW, U

    @abc.abstractmethod
    def _params_posterior(self, data, prior_mean, cache):
        'Laplace approximation of the parameters\' posteriors.'
        pass

    def _latents_posterior(self, psis, cache):
        prec, mean, W, WW = cache['prec'], cache['m'], cache['W'], cache['WW']
        cov = cache['latent_cov']
        psis_mean = psis - mean
        means = (prec * psis_mean @ W.t()) @ cov
        cache.update({
            'latent_means': means,
        })
        return cache

    def _precompute(self, data):
        prec, log_prec, m, mm, W, WW, WLW, U = self._extract_moments()
        dtype, device = m.dtype, m.device
        latent_prec = torch.eye(W.shape[0], dtype=dtype, device=device) + \
                      WLW
        latent_cov = torch.inverse(latent_prec)
        return {
            'log_prec': log_prec,
            'prec': prec,
            'm': m,
            'mm': mm,
            'W': W,
            'WW': WW,
            'WLW': WLW,
            'latent_cov': latent_cov,
            'U': U
        }

    def latent_posteriors(self, data, max_iter=20, conv_threshold=1e-3,
                          callback=None):
        '''Compute the latent posteriors for the given data.

        Args:
            data (``torch.Tensor``: Accumulated statistics of the
                likelihood functions and the the number of data points
                per function.
            max_iter (int): Maximum number of iterations.
            conv_threshold (float): Convergenge threshold.
            callback (func): Function called after each update
                (for debugging).

        Returns:
            A dictionary containing the parameters of the latent
            variables posterior and the intermediate computation.

        '''
        cache = self._precompute(data)
        prior_means = cache['m'][None, :]
        previous_quad_approx = float('-inf')
        for i in range(max_iter):
            cache = self._params_posterior(data, prior_means, cache)
            cache = self._latents_posterior(cache['params_mean'], cache)
            prior_means = cache['latent_means'] @ cache['W'] + cache['m']

            quad_approx = float(cache['quad_approx'].sum())
            diff = abs(quad_approx - previous_quad_approx)
            if diff <= conv_threshold:
                break
            if callback is not None:
                callback(previous_quad_approx, quad_approx)
            previous_quad_approx = quad_approx
        return cache

    def _compute_delta(self, cache):
        Wh = cache['latent_means'] @ cache['W']
        prior_means = Wh + cache['m']
        params_mean = cache['params_mean']
        latent_means = cache['latent_means']
        m = cache['m']
        mm = cache['mm']
        W = cache['W']
        h = cache['latent_means']
        hh = (cache['latent_cov'][None, :, :] + \
                  h[:, :, None] * h[:, None, :])
        U = cache['U']

        # WhhW
        idxs = tuple(range(U.shape[0]))
        I = torch.eye(len(m), dtype=m.dtype, device=m.device)
        ItrUhh = I[None] * torch.matmul(U[None], hh)[:, idxs, idxs].sum(dim=-1)[:, None, None]
        idxs = tuple(range(I.shape[0]))
        WhhW = ItrUhh + torch.matmul(torch.matmul(W.t()[None], hh), W[None])
        diag_WhhW = WhhW[:, idxs, idxs]

        delta = -.5 * cache['params_diag_m2'] + prior_means * params_mean
        delta -= .5 * (diag_WhhW + mm) + Wh * m
        cache.update({
            'Wh': Wh,
            'hh': hh,
        })
        return delta, cache

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return [[self.precision, self.mean, self.subspace, self.scale]]

    @staticmethod
    def sufficient_statistics(data):
        return data

    def expected_log_likelihood(self, stats, max_iter=20, conv_threshold=1e-3):
        cache = self.latent_posteriors(stats, max_iter, conv_threshold)
        delta, cache = self._compute_delta(cache)
        cache['delta'] = delta
        quad_approx = cache['quad_approx']
        latent_kl_div = kl_div_std_norm(cache['latent_means'],
                                        cache['latent_cov'])
        self.cache.update(cache)
        dim = len(cache['m'])
        param_kl_div = (-cache['prec'] * delta).sum(dim=-1) \
                       - .5 * cache['log_prec'].sum()
        param_kl_div += .5 * dim * math.log(2 * math.pi)
        param_kl_div -= cache['params_entropy']
        llh = quad_approx - param_kl_div
        return llh - latent_kl_div

    def accumulate(self, stats):
        # Mean stats.
        Wh = self.cache['Wh']
        prec = self.cache['prec']
        psi = self.cache['params_mean']
        mean_stats = torch.cat([
            prec * (psi - Wh).sum(dim=0),
            -.5 * len(Wh) * prec
        ], dim=-1)

        # Subspace stats.
        h = self.cache['latent_means']
        mean = self.cache['m']
        psi_mean = (psi - mean)
        psi_mean_L = h.t() @ (psi_mean * prec[None])
        M0, U0 = self.subspace.prior.to_std_parameters()
        U0_inv = torch.diag(self.scale.expected_value())
        acc_hh = self.cache['hh'].sum(dim=0)
        U_inv = U0_inv + (prec.sum() / len(mean)) * acc_hh
        vec_C = (U0_inv @ M0 + psi_mean_L)
        blocks = prec[:, None, None] * acc_hh
        idxs = list(range(h.shape[-1]))
        blocks[:, idxs, idxs] += self.scale.expected_value()
        blocks_inv = batch_inverse(blocks)
        M = (blocks_inv * vec_C.t()[:, None, :]).sum(dim=-1).t()
        nparams = torch.cat([
            (U_inv @ M).reshape(-1),
            -.5 * U_inv.reshape(-1)
        ])
        subspace_stats = nparams - self.subspace.prior.natural_parameters

        # Precision stats.
        delta = self.cache['delta']
        acc_delta = delta.sum(dim=0)
        s2 = .5 * len(h) * torch.ones_like(mean)
        dtype, device = mean.dtype, mean.device
        prec_stats = torch.cat([
            delta.sum(dim=0).reshape(-1),
            s2
        ])

        # Scale stats.
        dim = len(self.cache['m'])
        diag_WW = torch.diag(self.cache['WW'])
        diag_WM = torch.diag(self.cache['W'] @ M0.t() )
        diag_MM = torch.diag(M0 @ M0.t())
        dtype, device = diag_MM.dtype, diag_MM.device
        s2 = .5 * dim * torch.ones(h.shape[-1], dtype=dtype, device=device)
        scale_stats = torch.cat([
            -.5 * (diag_WW + diag_MM) + diag_WM,
            s2.reshape(-1)
        ])

        return {
            self.mean: mean_stats,
            self.subspace: subspace_stats,
            self.precision: prec_stats,
            self.scale: scale_stats
        }


class GeneralizedSubspaceModelFull(GeneralizedSubspaceModel):


    def _precompute(self, data):
        cache = super()._precompute(data)
        prec = cache['prec']
        dtype, device = cache['m'].dtype, cache['m'].device
        max_psis = self.llh_func.argmax(data)
        hessians = self.llh_func.hessian(max_psis, data, mode='full')
        hessian_psis = (hessians * max_psis[:, None, :]).sum(dim=-1)
        precs = torch.diag(prec) - hessians
        covs = batch_inverse(precs)
        logdets = torch.cat([_logdet(cov) for cov in covs])
        dim = max_psis.shape[-1]
        entropy = .5 * (logdets + dim * math.log(2 * math.pi) + dim)
        cache.update({
            'max_psis': max_psis,
            'hessians': hessians,
            'hessian_psis': hessian_psis,
            'params_cov': covs,
            'params_entropy': entropy.reshape(-1),
        })
        return cache

    def _params_posterior(self, data, prior_mean, cache):
        prior_prec = cache['prec']
        max_psis = cache['max_psis']
        hessians = cache['hessians']
        hessian_psis = cache['hessian_psis']
        covs = cache['params_cov']
        means = (covs * (prior_prec * prior_mean - \
                         hessian_psis)[:, None, :]).sum(dim=-1)

        psis_m2 = covs + means[:, :, None] * means[:, None, :]
        psisHpsis = torch.sum(hessians.reshape(len(max_psis), -1) * \
                              psis_m2.reshape(len(max_psis), -1), dim=-1)
        Hmax_psis = (hessians * max_psis[:, None, :]).sum(dim=-1)
        max_psisHmax_psis = (Hmax_psis * max_psis).sum(dim=-1)
        psisHmax_psis = (Hmax_psis * means).sum(dim=-1)
        quad_approx = self.llh_func(max_psis, data)
        quad_approx += .5 * (psisHpsis + max_psisHmax_psis) - psisHmax_psis
        idxs = tuple(range(covs.shape[-1]))
        psis_diag_m2 = psis_m2[:, idxs, idxs]
        cache.update({
            'params_mean': means,
            'params_diag_m2': psis_diag_m2,
            'quad_approx': quad_approx,
        })
        return cache


class GeneralizedSubspaceModelDiagonal(GeneralizedSubspaceModel):


    def _precompute(self, data):
        cache = super()._precompute(data)
        prec = cache['prec']
        dtype, device = cache['m'].dtype, cache['m'].device
        max_psis = self.llh_func.argmax(data)
        hessians = self.llh_func.hessian(max_psis, data, mode='diagonal')
        hessian_psis = hessians * max_psis
        precs = prec - hessians
        covs = 1/precs
        logdets = covs.log().sum(dim=-1)
        dim = max_psis.shape[-1]
        entropy = .5 * (logdets + dim * math.log(2 * math.pi) + dim)
        cache.update({
            'max_psis': max_psis,
            'hessians': hessians,
            'hessian_psis': hessian_psis,
            'params_cov': covs,
            'params_entropy': entropy.reshape(-1),
        })
        return cache

    def _params_posterior(self, data, prior_mean, cache):
        prior_prec = cache['prec']
        max_psis = cache['max_psis']
        hessians = cache['hessians']
        hessian_psis = cache['hessian_psis']
        covs = cache['params_cov']
        means = covs * (prior_prec * prior_mean - hessian_psis)
        psis_m2 = covs + means**2
        psisHpsis = torch.sum(hessians * psis_m2, dim=-1)
        Hmax_psis = hessians * max_psis
        max_psisHmax_psis = (Hmax_psis * max_psis).sum(dim=-1)
        psisHmax_psis = (Hmax_psis * means).sum(dim=-1)
        quad_approx = self.llh_func(max_psis, data)
        quad_approx += .5 * (psisHpsis + max_psisHmax_psis) - psisHmax_psis
        idxs = tuple(range(covs.shape[-1]))
        tr_m2 = covs.sum(dim=-1) + torch.sum(means**2, dim=-1)
        cache.update({
            'params_mean': means,
            'params_tr_m2': tr_m2,
            'quad_approx': quad_approx,
        })
        return cache


class GeneralizedSubspaceModelScalar(GeneralizedSubspaceModel):

    def _precompute(self, data):
        cache = super()._precompute(data)
        prec = cache['prec']
        dtype, device = cache['m'].dtype, cache['m'].device
        max_psis = self.llh_func.argmax(data)
        hessians = self.llh_func.hessian(max_psis, data, mode='scalar')
        hessian_psis = hessians[:, None] * max_psis
        precs = prec - hessians
        covs = 1/ precs
        dim = max_psis.shape[-1]
        logdets = dim * covs.log()
        entropy = .5 * (logdets + dim * math.log(2 * math.pi) + dim)
        cache.update({
            'max_psis': max_psis,
            'hessians': hessians,
            'hessian_psis': hessian_psis,
            'params_cov': covs,
            'params_entropy': entropy.reshape(-1),
        })
        return cache

    def _params_posterior(self, data, prior_mean, cache):
        prior_prec = cache['prec']
        max_psis = cache['max_psis']
        hessians = cache['hessians']
        hessian_psis = cache['hessian_psis']
        covs = cache['params_cov']
        means = covs[:, None] * (prior_prec * prior_mean - hessian_psis)
        psis_m2 = covs[:, None] + means**2
        psisHpsis = torch.sum(hessians[:, None] * psis_m2, dim=-1)
        Hmax_psis = hessians[:, None] * max_psis
        max_psisHmax_psis = (Hmax_psis * max_psis).sum(dim=-1)
        psisHmax_psis = (Hmax_psis * means).sum(dim=-1)
        quad_approx = self.llh_func(max_psis, data)
        quad_approx += .5 * (psisHpsis + max_psisHmax_psis) - psisHmax_psis
        idxs = tuple(range(covs.shape[-1]))
        tr_m2 = max_psis.shape[-1] * covs + torch.sum(means**2, dim=-1)
        cache.update({
            'params_mean': means,
            'params_tr_m2': tr_m2,
            'quad_approx': quad_approx,
        })
        return cache


__all__ = [
    'GeneralizedSubspaceModel',
]

