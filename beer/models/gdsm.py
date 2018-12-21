import abc
import math
import torch

from .bayesmodel import BayesianModel
from .parameters import BayesianParameter
from .parameters import BayesianParameterSet
from .parameters import ConstantParameter
from .gsm import kl_div_std_norm, batch_inverse
from .gsm import GeneralizedSubspaceModelFull
from .gsm import GeneralizedSubspaceModelDiagonal
from .gsm import GeneralizedSubspaceModelScalar
from ..priors import NormalFullCovariancePrior
from ..priors import NormalIsotropicCovariancePrior
from ..priors import GammaPrior
from ..priors import JointGammaPrior
from ..priors import HierarchicalMatrixNormalPrior
from ..priors import MatrixNormalPrior
from ..priors.wishart import _logdet
from ..utils import make_symposdef
from ..utils import onehot


class GeneralizedDiscriminantSubspaceModel(BayesianModel):
    'Bayesian Generalized Discriminant Subspace Model.'

    @staticmethod
    def create(llh_func, nclasses, mean_subspace, mean_dsubspace, global_mean,
               mean_prec, mean_scale, mean_dscale, noise_std=0.,
               prior_strength=1., hyper_prior_strength=1.,
               ds_hyper_prior_strength=1., hessian_type='full'):
        '''Create a Bayesian Generalized Subspace Model.

        Args:
            llh_func: (:any:`LikelihoodFunction`): The type of model
                of the subpsace.
            nclasses (int): Number of classes.
            mean_subspace (``torch.Tensor[K,D]``): Mean of the prior
                over the bases of the subspace. K is the dimension of
                the subspace and D the dimension of the parameter
                space.
            mean_dsubspace (``torch.Tensor[K2,D]``): Mean of the prior
                over the bases of the discriminant subspace.
                K2 is the dimension of the subspace and D the dimension
                of the parameter space.
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
            cls = GeneralizedDiscriminantSubspaceModelFull
        elif hessian_type == 'diagonal':
            cls = GeneralizedDiscriminantSubspaceModelDiagonal
        elif hessian_type == 'scalar':
            cls = GeneralizedDiscriminantSubspaceModelScalar
        else:
            raise ValueError(f'Unknown hessian type: "{hessian_type}"')

        dim_s = len(mean_subspace)
        dim_ds = len(mean_dsubspace)
        dim_o = len(global_mean)
        dtype, device = mean_subspace.dtype, mean_subspace.device
        with torch.no_grad():
            # Class means.
            mean = torch.zeros(dim_ds, dtype=dtype, device=device)
            cov = torch.eye(dim_ds, dtype=dtype, device=device)
            p_means = mean + noise_std * torch.randn(nclasses, dim_ds,
                                                     dtype=dtype, device=device)
            cmean_prior = NormalFullCovariancePrior(mean, cov)
            cmean_posteriors = [NormalFullCovariancePrior(p_means[i], cov)
                                for i in range(nclasses)]

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

            # Discriminant subspace scale prior/posterior.
            pad = torch.ones(dim_ds, dtype=dtype, device=device)
            shapes = ds_hyper_prior_strength * pad
            rates = (ds_hyper_prior_strength / mean_dscale) * pad
            dscale_prior = JointGammaPrior(shapes, rates)
            dscale_posterior = JointGammaPrior(shapes, rates)

            # Discriminant subspace prior/posterior.
            U = torch.diag(1/dscale_posterior.expected_value())
            r_M = mean_dsubspace + noise_std * torch.randn(dim_ds,
                                                           dim_o,
                                                           dtype=dtype,
                                                           device=device)
            dsubspace_prior = HierarchicalMatrixNormalPrior(mean_dsubspace,
                                                            dscale_posterior)
            dsubspace_posterior = MatrixNormalPrior(r_M, dsubspace_prior.cov)

            # Global mean prior/posterior.
            var = torch.tensor(1./prior_strength, dtype=dtype, device=device)
            mean_prior = NormalIsotropicCovariancePrior(global_mean, var)
            mean_posterior = NormalIsotropicCovariancePrior(global_mean, var)

            # Precision prior/posterior.
            shape = torch.tensor(prior_strength, dtype=dtype, device=device)
            rate =  torch.tensor(prior_strength / mean_prec, dtype=dtype,
                                 device=device)
            prec_prior = GammaPrior(shape, rate)
            prec_posterior = GammaPrior(shape, rate)

        return cls(llh_func, subspace_prior, subspace_posterior,
                   dsubspace_prior, dsubspace_posterior, mean_prior,
                   mean_posterior, cmean_prior, cmean_posteriors,
                   prec_prior, prec_posterior, scale_prior, scale_posterior,
                   dscale_prior, dscale_posterior)


    def __init__(self, llh_func, subspace_prior, subspace_posterior,
                 dsubspace_prior, dsubspace_posterior, mean_prior,
                 mean_posterior, cmean_prior, cmean_posteriors,
                 prec_prior, prec_posterior, scale_prior, scale_posterior,
                 dscale_prior, dscale_posterior):
        super().__init__()
        self.subspace = BayesianParameter(subspace_prior, subspace_posterior)
        self.dsubspace = BayesianParameter(dsubspace_prior, dsubspace_posterior)
        self.mean = BayesianParameter(mean_prior, mean_posterior)
        self.class_means = BayesianParameterSet(
            [BayesianParameter(cmean_prior, cmean_posteriors[i])
             for i in range(len(cmean_posteriors))]
        )
        self.precision = BayesianParameter(prec_prior, prec_posterior)
        self.scale = BayesianParameter(scale_prior, scale_posterior)
        self.ds_scale = BayesianParameter(dscale_prior, dscale_posterior)
        self.llh_func = llh_func
        self.scale.register_callback(self._update_subspace_prior)
        self.ds_scale.register_callback(self._update_dsubspace_prior)

    @property
    def nclasses(self):
        return len(self.class_means)

    def _update_subspace_prior(self):
        M, _ = self.subspace.prior.to_std_parameters()
        new_prior = HierarchicalMatrixNormalPrior(M, self.scale.posterior)
        self.subspace.prior = new_prior

    def _update_dsubspace_prior(self):
        M, _ = self.dsubspace.prior.to_std_parameters()
        new_prior = HierarchicalMatrixNormalPrior(M, self.ds_scale.posterior)
        self.dsubspace.prior = new_prior

    def _extract_moments(self):
        m, mm = self.mean.posterior.moments()
        W, WW = self.subspace.posterior.moments()
        S, SS = self.dsubspace.posterior.moments()
        prec, log_prec = self.precision.expected_natural_parameters()
        c_moments = [self.class_means[i].posterior.moments()
                     for i in range(self.nclasses)]
        cms = torch.cat([c_moments[i][0].reshape(1, -1)
                         for i in range(self.nclasses)], dim=0)
        cmms = torch.cat([c_moments[i][1].reshape(1, -1)
                          for i in range(self.nclasses)], dim=0)
        return prec, log_prec, m, mm, W, WW, S, SS, cms, cmms

    @abc.abstractmethod
    def _params_posterior(self, data, prior_mean, cache):
        'Laplace approximation of the parameters\' posteriors.'
        pass

    def _latents_posterior(self, psis, zSv, cache):
        prec, mean, W, WW = cache['prec'], cache['m'], cache['W'], cache['WW']
        cov = cache['latent_cov']
        psis_mean_Sv = psis - mean - zSv
        means = (prec * psis_mean_Sv @ W.t()) @ cov
        cache.update({
            'latent_means': means,
        })
        return cache

    def _precompute(self, data):
        prec, log_prec, m, mm, W, WW, S, SS, cms, cmms= self._extract_moments()
        dtype, device = m.dtype, m.device
        latent_prec = torch.eye(W.shape[0], dtype=dtype, device=device) + \
                      prec * WW
        latent_cov = torch.inverse(latent_prec)
        return {
            'log_prec': log_prec,
            'prec': prec,
            'm': m,
            'mm': mm,
            'W': W,
            'WW': WW,
            'S': S,
            'vec_SS': SS.reshape(-1),
            'v': cms,
            'vec_vv': cmms,
            'latent_cov': latent_cov,
            'Sv': cms @ S
        }

    def latent_posteriors(self, data, labels, max_iter=1000, conv_threshold=1e-12,
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
        oh_labels = onehot(labels, self.nclasses, data.dtype, data.device)
        #prior_means = cache['m'][None, :]
        prior_means = cache['max_psis']
        previous_quad_approx = float('-inf')
        for i in range(max_iter):
            zSv = oh_labels @ cache['Sv']
            cache = self._params_posterior(data, prior_means, cache)
            cache = self._latents_posterior(cache['params_mean'], zSv, cache)
            prior_means = cache['latent_means'] @ cache['W'] \
                          + zSv \
                          + cache['m'][None, :]

            quad_approx = float(cache['quad_approx'].sum())
            diff = abs(quad_approx - previous_quad_approx)
            if diff <= conv_threshold:
                break
            if callback is not None:
                callback(previous_quad_approx, quad_approx)
            previous_quad_approx = quad_approx
        cache['zSv'] = zSv
        cache['resps'] = oh_labels
        return cache

    def _compute_delta(self, cache):
        params_mean = cache['params_mean']
        latent_means = cache['latent_means']
        m = cache['m']
        mm = cache['mm']
        W = cache['W']
        zSv = cache['zSv']
        vec_vv = cache['vec_vv']
        vec_SS = cache['vec_SS']
        resps = cache['resps']
        z_vec_vv = resps @ vec_vv
        h = cache['latent_means']
        hh = (cache['latent_cov'][None, :, :] + \
                  h[:, :, None] * h[:, None, :])
        Wh = h @ W
        vec_hh = hh.reshape(len(h), -1)
        vec_WW = cache['WW'].reshape(-1)
        hWWh = vec_hh @ vec_WW
        zvSSv = z_vec_vv @ vec_SS
        prior_means = Wh + zSv + m

        delta = - .5 * (cache['params_tr_m2']  + hWWh + zvSSv + mm) \
                +  (prior_means * params_mean).sum(dim=-1) \
                - (Wh * m + (Wh + m) * zSv).sum(dim=-1)
        cache.update({
            'Wh': Wh,
            'hh': hh,
            'z_vec_vv': z_vec_vv
        })
        return delta, cache

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return [[self.mean], [self.subspace], [self.dsubspace],
                [*self.class_means], [self.precision],
                [self.scale, self.ds_scale]]

    @staticmethod
    def sufficient_statistics(data):
        return data

    def expected_log_likelihood(self, stats, labels, max_iter=100,
                                conv_threshold=1e-5):
        cache = self.latent_posteriors(stats, labels, max_iter,
                                       conv_threshold)
        delta, cache = self._compute_delta(cache)
        cache['delta'] = delta
        quad_approx = cache['quad_approx']
        latent_kl_div = kl_div_std_norm(cache['latent_means'],
                                        cache['latent_cov'])
        self.cache.update(cache)
        dim = len(cache['m'])
        param_kl_div = - cache['prec'] * delta \
                       - .5 * dim * cache['log_prec'] \
                       + .5 * dim * math.log(2 * math.pi)
        param_kl_div -= cache['params_entropy']
        llh = quad_approx - param_kl_div
        return llh - latent_kl_div

    def accumulate(self, stats):
        # Mean stats.
        Wh = self.cache['Wh']
        zSv = self.cache['zSv']
        prec = self.cache['prec']
        psi = self.cache['params_mean']
        mean_stats = torch.cat([
            (prec * (psi - Wh - zSv).sum(dim=0)).reshape(-1),
            (-.5 * len(Wh) * prec).reshape(-1)
        ], dim=-1)

        # Subspace stats.
        h = self.cache['latent_means']
        mean = self.cache['m']
        h_psi_mean = h.t() @ (psi - mean - zSv)
        acc_hh = self.cache['hh'].sum(dim=0)
        subspace_stats = torch.cat([
            (prec * h_psi_mean).reshape(-1),
            (-.5 * prec * acc_hh).reshape(-1)
        ])

        # Discriminant subspace stats.
        v = self.cache['v']
        z = self.cache['resps']
        acc_z_vec_vv = self.cache['z_vec_vv']
        psi_mean_Wh = (psi - mean - Wh)
        v_psi_mean_Wh = (z @ v).t() @ psi_mean_Wh
        dsubspace_stats = torch.cat([
            (prec * v_psi_mean_Wh).reshape(-1),
            -.5 * prec * acc_z_vec_vv.sum(dim=0)
        ])

        # Class means.
        vec_SS = self.cache['vec_SS']
        S = self.cache['S']
        psi_mean_Wh = (psi - mean - Wh)
        S_psi_mean_Wh = psi_mean_Wh @ S.t()
        class_means_stats = torch.cat([
           prec * z.t() @ S_psi_mean_Wh,
           -.5 * prec * z.sum(dim=0)[:, None] * vec_SS
        ], dim=-1)

        # Precision stats.
        delta = self.cache['delta']
        acc_delta = delta.sum(dim=0)
        s2 = torch.tensor(.5 * len(h) * len(mean),
                          dtype=mean.dtype, device=mean.device)
        dtype, device = mean.dtype, mean.device
        prec_stats = torch.cat([
            delta.sum(dim=0).reshape(-1),
            s2.reshape(-1)
        ])

        # Scale stats.
        dim = len(self.cache['m'])
        M0, _ = self.subspace.prior.to_std_parameters()
        diag_WW = torch.diag(self.cache['WW'])
        diag_WM = torch.diag(self.cache['W'] @ M0.t() )
        diag_MM = torch.diag(M0 @ M0.t())
        dtype, device = diag_MM.dtype, diag_MM.device
        s2 = .5 * dim * torch.ones(h.shape[-1], dtype=dtype, device=device)
        scale_stats = torch.cat([
            -.5 * (diag_WW + diag_MM) + diag_WM,
            s2.reshape(-1)
        ])

        # Scale stats.
        M0, _ = self.dsubspace.prior.to_std_parameters()
        ds_dim = v.shape[-1]
        diag_SS = self.cache['vec_SS'].reshape(ds_dim, ds_dim).diag()
        diag_SM = (self.cache['S'] @ M0.t()).diag()
        diag_MM = (M0 @ M0.t()).diag()
        s2 = .5 * dim * torch.ones(ds_dim, dtype=dtype, device=device)
        ds_scale_stats = torch.cat([
            -.5 * (diag_SS + diag_MM) + diag_SM,
            s2.reshape(-1)
        ])

        return {
            self.mean: mean_stats,
            self.subspace: subspace_stats,
            self.dsubspace: dsubspace_stats,
            self.precision: prec_stats,
            self.scale: scale_stats,
            self.ds_scale: ds_scale_stats,
            **dict(zip(self.class_means, class_means_stats))
        }


class GeneralizedDiscriminantSubspaceModelFull(GeneralizedDiscriminantSubspaceModel):


    def _precompute(self, data):
        cache = super()._precompute(data)
        prec = cache['prec']
        dtype, device = cache['m'].dtype, cache['m'].device
        max_psis = self.llh_func.argmax(data)
        hessians = self.llh_func.hessian(max_psis, data, mode='full')
        hessian_psis = (hessians * max_psis[:, None, :]).sum(dim=-1)
        I = torch.eye(hessians.shape[-1], dtype=dtype, device=device)
        precs = (prec * I)[None] - hessians
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
        psis_tr_m2 = psis_m2[:, idxs, idxs].sum(dim=-1)
        cache.update({
            'params_mean': means,
            'params_tr_m2': psis_tr_m2,
            'quad_approx': quad_approx,
        })
        return cache


class GeneralizedDiscriminantSubspaceModelDiagonal(GeneralizedDiscriminantSubspaceModel):


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
        max_psisHmax_psis = (hessian_psis * max_psis).sum(dim=-1)
        psisHmax_psis = (hessian_psis * means).sum(dim=-1)
        quad_approx = self.llh_func(max_psis, data)
        quad_approx += .5 * (psisHpsis + max_psisHmax_psis) - psisHmax_psis
        psis_tr_m2 = (covs + means**2).sum(dim=-1)
        cache.update({
            'params_mean': means,
            'params_tr_m2': psis_tr_m2,
            'quad_approx': quad_approx,
        })
        return cache


class GeneralizedDiscriminantSubspaceModelScalar(GeneralizedDiscriminantSubspaceModel):

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
        psis_tr_m2 = covs + torch.sum(means**2, dim=-1)
        cache.update({
            'params_mean': means,
            'params_tr_m2': psis_tr_m2,
            'quad_approx': quad_approx,
        })
        return cache


__all__ = [
    'GeneralizedDiscriminantSubspaceModel',
]

