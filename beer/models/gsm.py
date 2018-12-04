import abc
import math
import torch

from .bayesmodel import BayesianModel
from .parameters import BayesianParameter, ConstantParameter
from ..priors import NormalFullCovariancePrior
from ..priors import GammaPrior
from ..priors import JointGammaPrior
from ..priors import MatrixNormalPrior
from ..priors.wishart import _logdet
from ..utils import make_symposdef


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
               noise_std=0., prior_strength=1., hessian_type='full'):
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
            mean_prec (``torch.Tensor[0]``): Mean of the prior over
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
        args = (
            llh_func, mean_subspace, global_mean, mean_prec, mean_scale,
            noise_std, prior_strength
        )
        if hessian_type == 'full':
            return GeneralizedSubspaceModelFull.create(*args)
        elif hessian_type == 'diagonal':
            return GeneralizedSubspaceModelDiagonal.create(*args)
        elif hessian_type == 'scalar':
            return GeneralizedSubspaceModelScalar.create(*args)
        else:
            raise ValueError(f'Unknown hessian type: "{hessian_type}"')


    def __init__(self, llh_func, subspace_prior, subspace_posterior,
                 mean_prior, mean_posterior, prec_prior, prec_posterior,
                 scale_prior, scale_posterior):
        super().__init__()
        self.subspace = BayesianParameter(subspace_prior, subspace_posterior)
        self.mean = BayesianParameter(mean_prior, mean_posterior)
        self.precision = BayesianParameter(prec_prior, prec_posterior)
        self.scale = BayesianParameter(scale_prior, scale_posterior)
        self.llh_func = llh_func

    def _extract_moments(self):
        prec = self.precision.expected_value()
        m, mm = self.mean.posterior.moments()
        W, WW = self.subspace.posterior.moments()
        return prec, m, mm, W, WW

    @abc.abstractmethod
    def _params_posterior(self, data, prior_mean, cache):
        'Laplace approximation of the parameters\' posteriors.'
        pass

    @abc.abstractmethod
    def _latents_posterior(self, expected_params, cache):
        'Posterior over the latent variable the "i-vector".'
        pass

    @abc.abstractmethod
    def latent_posteriors(self, data, max_iter=20, conv_threshold=1e-3):
        '''Compute the latent posteriors for the given data.

        Args:
            data (``torch.Tensor``: Accumulated statistics of the
                likelihood functions and the the number of data points
                per function.
            max_iter (int): Maximum number of iterations.
            conv_threshold (float): Convergenge threshold.

        Returns:
            A dictionary containing the parameters of the latent
            variables posterior and the intermediate computation.

        '''
        cache = self._precompute(data)
        prior_means = cache['m'][None, :]
        previous_quad_approx = float('-inf')
        for i in range(max_iter):
            cache = self._params_posterior(data.double(), prior_means, cache)
            cache = self._latents_posterior(cache['params_mean'], cache)
            prior_means = cache['latent_means'] @ cache['W'] + cache['m']

            quad_approx = float(cache['quad_approx'].sum())
            if abs(quad_approx - previous_quad_approx) <= conv_threshold:
                break
            previous_quad_approx = quad_approx
        return cache

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return [[self.mean], [self.subspace]]

    @staticmethod
    def sufficient_statistics(data):
        return data

    def expected_log_likelihood(self, stats):
        l_posts, cache = self.latent_posteriors(stats, cache=True)
        cache.update(self._params_posterior())
        quad_approx = cache['quad_approx']
        param_entropy = cache['param_entropy']
        param_llh = cache['param_llh']
        param_kl_div = param_llh - param_entropy
        latent_kl_div = cache['latent_kl_div']
        return quad_approx - param_kl_div -latent_kl_div

    @abc.abstractmethod
    def accumulate(self, stats):
        pass


class GeneralizedSubspaceModelFull(GeneralizedSubspaceModel):

    @classmethod
    def create(cls, llh_func, mean_subspace, global_mean, mean_prec,
               mean_scale, noise_std=0., prior_strength=1.):
        dim_s = len(mean_subspace)
        dim_o = len(global_mean)
        dtype, device = mean_subspace.dtype, mean_subspace.device
        with torch.no_grad():
            # Subspace prior/posterior.
            mean_subspace.requires_grad = False
            global_mean.requires_grad = False
            U = torch.eye(dim_s, dtype=dtype, device=device)
            r_M = mean_subspace + noise_std * torch.randn(dim_s,
                                                          dim_o,
                                                          dtype=dtype,
                                                          device=device)
            subspace_prior = MatrixNormalPrior(mean_subspace, U)
            subspace_posterior = MatrixNormalPrior(r_M, U)

            # Global mean prior/posterior.
            S = prior_strength * torch.eye(dim_o, dim_o, dtype=dtype,
                                           device=device)
            mean_prior = NormalFullCovariancePrior(global_mean, S)
            mean_posterior = NormalFullCovariancePrior(global_mean, S)

            # Precision prior/posterior.
            shape = torch.tensor(prior_strength, dtype=dtype, device=device)
            rate =  torch.tensor(prior_strength / mean_prec, dtype=dtype,
                                 device=device)
            prec_prior = GammaPrior(shape, rate)
            prec_posterior = GammaPrior(shape, rate)

            # Scale prior/posterior.
            shapes = prior_strength * torch.ones(dim_s, dtype=dtype,
                                                 device=device)
            rates = (prior_strength / mean_scale) * torch.ones(dim_s,
                                                               dtype=dtype,
                                                               device=device)
            scale_prior = JointGammaPrior(shapes, rates)
            scale_posterior = JointGammaPrior(shapes, rates)

            return cls(llh_func, subspace_prior, subspace_posterior,
                       mean_prior, mean_posterior, prec_prior, prec_posterior,
                       scale_prior, scale_posterior)

    def _precompute(self, data):
        prec, m, mm, W, WW = self._extract_moments()
        dtype, device = m.dtype, m.device
        max_psis = self.llh_func.argmax(data)
        hessians = self.llh_func.hessian(max_psis, data, mode='full')
        hessian_psis = (hessians * max_psis[:, None, :]).sum(dim=-1)
        precs = prec * torch.eye(max_psis.shape[-1], dtype=dtype,
                                 device=device) - hessians
        covs = batch_inverse(precs)
        logdets = torch.cat([_logdet(cov) for cov in covs])
        dim = max_psis.shape[-1]
        entropy = .5 * (logdets + dim * math.log(2 * math.pi) + dim)

        latent_prec = torch.eye(W.shape[0], dtype=dtype, device=device) + \
                      prec * WW
        latent_cov = torch.inverse(latent_prec)

        return {
            'prec': prec,
            'max_psis': max_psis,
            'hessians': hessians,
            'hessian_psis': hessian_psis,
            'params_cov': covs,
            'entropy': entropy,
            'm': m,
            'mm': mm,
            'W': W,
            'WW': WW,
            'latent_cov': latent_cov,
        }

    def _params_posterior(self, data, prior_mean, cache):
        prior_prec = cache['prec']
        max_psis = cache['max_psis']
        hessians = cache['hessians']
        hessian_psis = cache['hessian_psis']
        covs = cache['params_cov']
        means = (covs * (prior_prec * prior_mean - \
                         hessian_psis)[:, None, :]).sum(dim=-1)
        psi_maxpsi = (prior_mean - max_psis)
        Hpsi_maxpsi = (hessians * psi_maxpsi[:, None, :]).sum(dim=-1)
        quad_approx = self.llh_func(max_psis, data) + \
                      .5 * torch.sum(Hpsi_maxpsi * psi_maxpsi, dim=-1)
        idxs = tuple(range(covs.shape[-1]))
        tr_m2 = covs[:, idxs, idxs].sum(dim=-1) + torch.sum(means**2, dim=-1)
        cache.update({
            'params_mean': means,
            'params_tr_m2': tr_m2,
            'quad_approx': quad_approx,
        })
        return cache

    def _latents_posterior(self, psis, cache):
        prec, mean, W, WW = cache['prec'], cache['m'], cache['W'], cache['WW']
        cov = cache['latent_cov']
        psis_mean = psis - mean
        means = (prec * psis_mean @ W.t()) @ cov
        cache.update({
            'latent_means': means,
        })
        return cache

    def _quad_approx(self, cache):
        opt, hessians = cache['opt'], cache['hessians']
        m, mm = cache['m'], cache['mm']
        W, WW, U = cache['W'], cache['WW'], cache['U']
        WHW = cache['WHW']
        H, HH = cache['H'], cache['HH']
        stats = cache['stats']

        mHm = torch.sum(hessians.reshape(len(hessians), -1) * mm, dim=-1)
        hW = H @ W
        Hm = hessians @ m
        mHWh = (hW * Hm).sum(dim=-1)
        oHm = torch.sum(opt * Hm, dim=-1)
        oH = (hessians * opt[:, None, :]).sum(dim=-1)
        oHWh = torch.sum(oH * hW, dim=-1)
        hWHWh = (HH * WHW).sum(dim=-1)

        opt_quad = opt[:, :, None] * opt[:, None, :]
        opt_quad = opt_quad.reshape(len(opt), -1)
        rs_hessians = hessians.reshape(len(opt), -1)
        quad_opt = torch.sum(rs_hessians * opt_quad, dim=-1)

        new_cache = {**cache,
            'hW': hW,
        }

        return  self.llh_func(opt, stats) \
                     + .5 * (hWHWh + mHm + quad_opt  \
                            + 2 * (mHWh - oHm - oHWh)), new_cache

        nparams_0 = self.latent_prior.value.natural_parameters
        for i, l_post in enumerate(l_posts):
            l_post.natural_parameters = nparams_0 + acc_stats[i]

        if not cache:
            return l_posts

        # TODO: possible optimization by doing a single for-loop.
        H = torch.cat([pdf.moments()[0][None, :]
                       for pdf in l_posts], dim=0)
        HH = torch.cat([pdf.moments()[1][None, :]
                        for pdf in l_posts], dim=0)
        cache = {
            'U': U,
            'opt': opt,
            'hessians': hessians,
            'tr_hessians': tr_hessians,
            'm': m,
            'mm': mm,
            'W': W,
            'WW': WW,
            'H': H,
            'HH': HH,
            'stats': stats,
            'WHW': WHW,
        }
        return l_posts, cache

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def accumulate(self, stats):
        hessians = self.cache['hessians']
        tr_hessians = self.cache['tr_hessians']
        hW = self.cache['hW']
        opt = self.cache['opt']
        m = self.cache['m']
        H = self.cache['H']
        HH = self.cache['HH']

        # Mean stats.
        HWh_o = torch.sum(hessians * (opt - hW)[:, None, :], dim=-1)
        #sum_hessians = make_symposdef(.5 * hessians.sum(dim=0))
        sum_hessians = .5 * hessians.sum(dim=0)
        mean_stats = torch.cat([
            -HWh_o.sum(dim=0),
            sum_hessians.reshape(-1)
        ], dim=-1)

        # Subspace stats.
        #idxs = tuple(range(hessians.shape[-1]))
        #isometric_params = hessians[:, idxs, idxs].max(dim=-1)[0][:, None]
        isometric_params = tr_hessians[:, None] / len(m)
        to_m = (opt - m) * isometric_params
        tHH = HH * isometric_params
        #tHH = make_symposdef(.5 * tHH.sum(dim=0).reshape(HH.shape[-1], -1))
        tHH = .5 * tHH.sum(dim=0).reshape(HH.shape[-1], -1)
        subspace_stats = torch.cat([
            -(H.t() @ to_m).reshape(-1),
            tHH.reshape(-1)
        ])

        return {
            self.mean: mean_stats,
            self.subspace: subspace_stats
        }


class GeneralizedSubspaceModelDiagonal(GeneralizedSubspaceModel):

    @classmethod
    def create(cls, llh_func, mean_subspace, global_mean, noise_std=0.,
               prior_strength=1.):
        dim_s = len(mean_subspace)
        dim_o = len(global_mean)
        dtype, device = mean_subspace.dtype, mean_subspace.device
        with torch.no_grad():
            # Subspace prior/posterior.
            mean_subspace.requires_grad = False
            global_mean.requires_grad = False
            U = torch.eye(dim_s, dtype=dtype, device=device)
            r_M = mean_subspace + noise_std * torch.randn(dim_s,
                                                          dim_o,
                                                          dtype=dtype,
                                                          device=device)
            subspace_prior = MatrixNormalPrior(mean_subspace, U)
            subspace_posterior = MatrixNormalPrior(r_M, U)

            # Global mean prior/posterior.
            S = prior_strength * torch.eye(dim_o, dim_o, dtype=dtype,
                                           device=device)
            mean_prior = NormalFullCovariancePrior(global_mean, S)
            mean_posterior = NormalFullCovariancePrior(global_mean, S)

            # Latent prior.
            latent_mean = torch.zeros(dim_s, dtype=dtype, device=device)
            latent_cov = torch.eye(dim_s, dim_s, dtype=dtype, device=device)
            latent_prior = NormalFullCovariancePrior(latent_mean, latent_cov)

            return cls(llh_func, subspace_prior, subspace_posterior,
                       mean_prior, mean_posterior, latent_prior)

    def _quad_approx(self, cache):
        opt, hessians = cache['opt'], cache['hessians']
        m, mm = cache['m'], cache['mm']
        W, WW, U = cache['W'], cache['WW'], cache['U']
        WHW = cache['WHW']
        H, HH = cache['H'], cache['HH']
        stats = cache['stats']

        mm_diag = torch.diag(mm.reshape(m.shape[0], m.shape[0]))
        mHm = torch.sum(hessians * mm_diag[None], dim=-1)
        hW = H @ W
        Hm = hessians * m
        mHWh = (hW * Hm).sum(dim=-1)
        oHm = torch.sum(opt * Hm, dim=-1)
        oH = hessians * opt
        oHWh = torch.sum(oH * hW, dim=-1)
        hWHWh = (HH * WHW).sum(dim=-1)
        quad_opt = torch.sum((hessians * opt) * opt, dim=-1)

        new_cache = {**cache,
            'hW': hW,
        }

        return  self.llh_func(opt, stats) \
                     + .5 * (hWHWh + mHm + quad_opt  \
                            + 2 * (mHWh - oHm - oHWh)), new_cache

    def latent_posteriors(self, stats, cache=False):
        l_posts = self._create_latent_posteriors(len(stats))
        m, mm, W, WW = self._extract_moments()

        opt = self.llh_func.argmax(stats)
        hessians = self.llh_func.hessian(opt, stats, mode='diagonal')

        U = self.subspace.posterior.cov
        tr_hessians = hessians.sum(dim=-1)
        HW = hessians[:, None, :] * W[None]
        WHW = tr_hessians[:, None, None] * U + HW @ W.t()
        WHW = WHW.reshape(len(hessians), -1)
        m_opt = (opt - m[None])
        Hm_o = hessians * m_opt
        acc_stats =  torch.cat([
            -Hm_o @ W.t(),
            .5 * WHW
        ], dim=-1)

        nparams_0 = self.latent_prior.value.natural_parameters
        for i, l_post in enumerate(l_posts):
            l_post.natural_parameters = nparams_0 + acc_stats[i]

        if not cache:
            return l_posts

        # TODO: possible optimization by doing a single for-loop.
        H = torch.cat([pdf.moments()[0][None, :]
                       for pdf in l_posts], dim=0)
        HH = torch.cat([pdf.moments()[1][None, :]
                        for pdf in l_posts], dim=0)
        cache = {
            'U': U,
            'opt': opt,
            'hessians': hessians,
            'tr_hessians': tr_hessians,
            'm': m,
            'mm': mm,
            'W': W,
            'WW': WW,
            'H': H,
            'HH': HH,
            'stats': stats,
            'WHW': WHW,
        }
        return l_posts, cache

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def accumulate(self, stats):
        hessians = self.cache['hessians']
        tr_hessians = self.cache['tr_hessians']
        hW = self.cache['hW']
        opt = self.cache['opt']
        m = self.cache['m']
        H = self.cache['H']
        HH = self.cache['HH']

        # Mean stats.
        HWh_o = hessians * (opt - hW)
        I = torch.eye(len(m), dtype=m.dtype, device=m.device)
        mean_stats = torch.cat([
            -HWh_o.sum(dim=0),
            .5 * (hessians[:, None, :] * I[None]).sum(dim=0).reshape(-1)
        ], dim=-1)

        # Subspace stats.
        #isometric_params = hessians.max(dim=-1)[0][:, None]
        isometric_params = tr_hessians[:, None] / len(m)
        to_m = (opt - m) * isometric_params
        tHH = HH * isometric_params
        subspace_stats = torch.cat([
            -(H.t() @ to_m).reshape(-1),
            .5 * tHH.sum(dim=0)
        ])

        return {
            self.mean: mean_stats,
            self.subspace: subspace_stats
        }


class GeneralizedSubspaceModelScalar(GeneralizedSubspaceModel):

    @classmethod
    def create(cls, llh_func, mean_subspace, global_mean, noise_std=0.,
               prior_strength=1.):
        dim_s = len(mean_subspace)
        dim_o = len(global_mean)
        dtype, device = mean_subspace.dtype, mean_subspace.device
        with torch.no_grad():
            # Subspace prior/posterior.
            mean_subspace.requires_grad = False
            global_mean.requires_grad = False
            U = torch.eye(dim_s, dtype=dtype, device=device)
            r_M = mean_subspace + noise_std * torch.randn(dim_s,
                                                          dim_o,
                                                          dtype=dtype,
                                                          device=device)
            subspace_prior = MatrixNormalPrior(mean_subspace, U)
            subspace_posterior = MatrixNormalPrior(r_M, U)

            # Global mean prior/posterior.
            S = prior_strength * torch.eye(dim_o, dim_o, dtype=dtype,
                                           device=device)
            mean_prior = NormalFullCovariancePrior(global_mean, S)
            mean_posterior = NormalFullCovariancePrior(global_mean, S)

            # Latent prior.
            latent_mean = torch.zeros(dim_s, dtype=dtype, device=device)
            latent_cov = torch.eye(dim_s, dim_s, dtype=dtype, device=device)
            latent_prior = NormalFullCovariancePrior(latent_mean, latent_cov)

            return cls(llh_func, subspace_prior, subspace_posterior,
                       mean_prior, mean_posterior, latent_prior)

    def _quad_approx(self, cache):
        opt, hessians = cache['opt'], cache['hessians']
        m, mm = cache['m'], cache['mm']
        W, WW, U = cache['W'], cache['WW'], cache['U']
        WHW = cache['WHW']
        H, HH = cache['H'], cache['HH']
        stats = cache['stats']

        mm_diag = torch.diag(mm.reshape(m.shape[0], m.shape[0]))
        mHm = torch.sum(hessians[:, None] * mm_diag[None], dim=-1)
        hW = H @ W
        Hm = hessians[:,None] * m
        mHWh = (hW * Hm).sum(dim=-1)
        oHm = torch.sum(opt * Hm, dim=-1)
        oH = hessians[:, None] * opt
        oHWh = torch.sum(oH * hW, dim=-1)
        hWHWh = (HH * WHW).sum(dim=-1)
        quad_opt = torch.sum((hessians[:, None] * opt) * opt, dim=-1)

        new_cache = {**cache,
            'hW': hW,
        }

        return  self.llh_func(opt, stats) \
                     + .5 * (hWHWh + mHm + quad_opt  \
                            + 2 * (mHWh - oHm - oHWh)), new_cache

    def latent_posteriors(self, stats, cache=False):
        l_posts = self._create_latent_posteriors(len(stats))
        m, mm, W, WW = self._extract_moments()

        opt = self.llh_func.argmax(stats)
        hessians = self.llh_func.hessian(opt, stats, mode='scalar')

        U = self.subspace.posterior.cov
        tr_hessians = hessians * len(m)
        HW = hessians[:, None, None] * W[None]
        WHW = tr_hessians[:, None, None] * U + HW @ W.t()
        WHW = WHW.reshape(len(hessians), -1)
        m_opt = (opt - m[None])
        Hm_o = hessians[:, None] * m_opt
        acc_stats =  torch.cat([
            -Hm_o @ W.t(),
            .5 * WHW
        ], dim=-1)

        nparams_0 = self.latent_prior.value.natural_parameters
        for i, l_post in enumerate(l_posts):
            l_post.natural_parameters = nparams_0 + acc_stats[i]

        if not cache:
            return l_posts

        # TODO: possible optimization by doing a single for-loop.
        H = torch.cat([pdf.moments()[0][None, :]
                       for pdf in l_posts], dim=0)
        HH = torch.cat([pdf.moments()[1][None, :]
                        for pdf in l_posts], dim=0)
        cache = {
            'U': U,
            'opt': opt,
            'hessians': hessians,
            'tr_hessians': tr_hessians,
            'm': m,
            'mm': mm,
            'W': W,
            'WW': WW,
            'H': H,
            'HH': HH,
            'stats': stats,
            'WHW': WHW,
        }
        return l_posts, cache

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def accumulate(self, stats):
        hessians = self.cache['hessians']
        tr_hessians = self.cache['tr_hessians']
        hW = self.cache['hW']
        opt = self.cache['opt']
        m = self.cache['m']
        H = self.cache['H']
        HH = self.cache['HH']

        # Mean stats.
        HWh_o = hessians[:, None] * (opt - hW)
        I = torch.eye(len(m), dtype=m.dtype, device=m.device)
        mean_stats = torch.cat([
            -HWh_o.sum(dim=0),
            .5 * (hessians[:, None, None] * I[None]).sum(dim=0).reshape(-1)
        ], dim=-1)

        # Subspace stats.
        isometric_params = tr_hessians[:, None] / len(m)
        to_m = (opt - m) * isometric_params
        tHH = HH * isometric_params
        subspace_stats = torch.cat([
            -(H.t() @ to_m).reshape(-1),
            .5 * tHH.sum(dim=0)
        ])

        return {
            self.mean: mean_stats,
            self.subspace: subspace_stats
        }


__all__ = [
    'GeneralizedSubspaceModel',
]
