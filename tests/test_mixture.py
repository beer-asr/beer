'Test the Mixture model.'


import sys
sys.path.insert(0, './')
import unittest
import numpy as np
from scipy.special import logsumexp
import math
import beer
import torch


TOLPLACES = 5
TOL = 10 ** (-TOLPLACES)


class TestMixtureDiag(unittest.TestCase):

    def setUp(self):
        self.p_mean = torch.zeros(2)
        self.p_cov = torch.eye(2) * 2
        self.args = {
            'prior_mean': self.p_mean,
            'prior_cov': self.p_cov,
            'prior_count':1,
            'random_init':True
        }

    def test_create(self):
        gmm_diag = beer.Mixture.create(10,
            beer.NormalDiagonalCovariance.create, self.args, prior_count=1.)
        self.assertTrue(np.allclose(gmm_diag.weights.numpy(),
            .1 * np.ones(10)))

    def test_sufficient_statistics(self):
        gmm_diag = beer.Mixture.create(10,
            beer.NormalDiagonalCovariance.create, self.args, prior_count=1.)
        X = torch.ones(10, 2)
        s1 = gmm_diag.sufficient_statistics(X)
        s2 = beer.NormalDiagonalCovariance.sufficient_statistics(X)
        self.assertTrue(np.allclose(s1.numpy()[:, :-1], s2.numpy()))
        self.assertTrue(np.allclose(s1.numpy()[:, -1], np.ones(10)))

    def test_exp_llh(self):
        gmm_diag = beer.Mixture.create(10,
            beer.NormalDiagonalCovariance.create, self.args, prior_count=1.)
        X = torch.ones(20, 2)
        np_params_matrix = gmm_diag._np_params_matrix.numpy()
        T = gmm_diag.sufficient_statistics(X).numpy()
        per_component_exp_llh = T @ np_params_matrix.T
        exp_llh1 = logsumexp(per_component_exp_llh, axis=1)
        resps = np.exp(per_component_exp_llh - exp_llh1[:, None])
        exp_llh1 -= .5 * X.shape[1] * np.log(2 * np.pi)
        acc_stats1 = resps.T @ T[:, :-1], resps.sum(axis=0)

        exp_llh2, acc_stats2 = gmm_diag.exp_llh(X, accumulate=True)
        self.assertTrue(np.allclose(exp_llh1, exp_llh2.numpy(), rtol=TOL,
            atol=TOL))

    def test_kl_div_posterior_prior(self):
        args = {
            'prior_mean': self.p_mean,
            'prior_cov': self.p_cov,
            'prior_count':1,
            'random_init':False
        }
        gmm_diag = beer.Mixture.create(10,
            beer.NormalDiagonalCovariance.create, args, prior_count=1.)
        self.assertAlmostEqual(gmm_diag.kl_div_posterior_prior(), 0.)

        args = {
            'prior_mean': self.p_mean,
            'prior_cov': self.p_cov,
            'prior_count':1,
            'random_init':True
        }
        gmm_diag = beer.Mixture.create(10,
            beer.NormalDiagonalCovariance.create, args, prior_count=1.)
        self.assertGreater(gmm_diag.kl_div_posterior_prior(), 0.)

    def test_natural_grad_update(self):
        gmm_diag = beer.Mixture.create(10,
            beer.NormalDiagonalCovariance.create, self.args, prior_count=1.)
        X = torch.ones(20, 2)
        np_params_matrix = gmm_diag._np_params_matrix.numpy()
        T = gmm_diag.sufficient_statistics(X).numpy()
        per_component_exp_llh = T @ np_params_matrix.T
        exp_llh1 = logsumexp(per_component_exp_llh, axis=1)
        resps = np.exp(per_component_exp_llh - exp_llh1[:, None])
        exp_llh1 -= .5 * X.shape[1] * np.log(2 * np.pi)
        acc_stats1 = resps.T @ T[:, :-1], resps.sum(axis=0)
        p_nparams = gmm_diag.prior_weights.natural_params.numpy()
        nparams1 = gmm_diag.posterior_weights.natural_params.numpy().copy()
        grad = p_nparams + .5 * acc_stats1[1] - nparams1
        nparams1 += .1 * grad

        exp_llh2, acc_stats2 = gmm_diag.exp_llh(X, accumulate=True)
        gmm_diag.natural_grad_update(acc_stats2, .5, .1)
        nparams2 = gmm_diag.posterior_weights.natural_params.numpy()

        self.assertTrue(np.allclose(nparams1, nparams2, rtol=TOL, atol=TOL))


if __name__ == '__main__':
    unittest.main()

