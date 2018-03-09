'Test the Mixture model.'


import sys
sys.path.insert(0, './')
import unittest
import numpy as np
from scipy.special import logsumexp
import math
import beer
import torch


TOLPLACES = 4
TOL = 10 ** (-TOLPLACES)


class TestMixture:

    def test_create(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        self.assertTrue(np.allclose(model.weights.numpy(),
            (1./len(self.prior_counts)) * np.ones(len(self.prior_counts))))

    def test_sufficient_statistics(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        s1 = model.sufficient_statistics(self.X)
        s2 = self.comp_type.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1.numpy()[:, :-1], s2.numpy()))
        self.assertTrue(np.allclose(s1.numpy()[:, -1], np.ones(self.X.size(0))))

    def test_exp_llh(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        np_params_matrix = model._np_params_matrix.numpy()
        T = model.sufficient_statistics(self.X).numpy()
        per_component_exp_llh = T @ np_params_matrix.T
        exp_llh1 = logsumexp(per_component_exp_llh, axis=1)
        resps = np.exp(per_component_exp_llh - exp_llh1[:, None])
        exp_llh1 -= .5 * self.X.shape[1] * math.log(2 * math.pi)
        acc_stats1 = resps.T @ T[:, :-1], resps.sum(axis=0)
        exp_llh2, acc_stats2 = model.exp_llh(self.X, accumulate=True)
        exp_llh2 = exp_llh2.numpy()
        self.assertTrue(np.allclose(exp_llh1.astype(exp_llh2.dtype), exp_llh2,
             atol=TOL))

    def test_kl_div_posterior_prior(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        if self.args['random_init'] == False:
            self.assertAlmostEqual(model.kl_div_posterior_prior(), 0.)
        else:
            self.assertGreater(model.kl_div_posterior_prior(), 0.)

    def test_natural_grad_update(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        np_params_matrix = model._np_params_matrix.numpy()
        T = model.sufficient_statistics(self.X).numpy()
        per_component_exp_llh = T @ np_params_matrix.T
        exp_llh1 = logsumexp(per_component_exp_llh, axis=1)
        resps = np.exp(per_component_exp_llh - exp_llh1[:, None])
        exp_llh1 -= .5 * self.X.shape[1] * np.log(2 * np.pi)
        acc_stats1 = resps.T @ T[:, :-1], resps.sum(axis=0)
        p_nparams = model.prior_weights.natural_params.numpy()
        nparams1 = model.posterior_weights.natural_params.numpy().copy()
        grad = p_nparams + .5 * acc_stats1[1] - nparams1
        nparams1 += .1 * grad
        exp_llh2, acc_stats2 = model.exp_llh(self.X, accumulate=True)
        model.natural_grad_update(acc_stats2, .5, .1)
        nparams2 = model.posterior_weights.natural_params.numpy()
        self.assertTrue(np.allclose(nparams1, nparams2, atol=TOL))

torch.manual_seed(10)
data = torch.randn(20, 2)

gmm_diag1F = {
    'X': data.float(),
    'prior_counts': torch.ones(10).float(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).float(),
        'prior_cov': torch.eye(2).float(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_diag1D = {
    'X': data.double(),
    'prior_counts': torch.ones(10).double(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_diag2F = {
    'X': data.float(),
    'prior_counts': torch.ones(1).float(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).float(),
        'prior_cov': torch.eye(2).float(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_diag2D = {
    'X': data.double(),
    'prior_counts': torch.ones(1).double(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_diag3F = {
    'X': data.float(),
    'prior_counts': 1,
    'prior_counts': torch.ones(2).float(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).float(),
        'prior_cov': torch.eye(2).float(),
        'prior_count': 1,
        'random_init': True
    }
}

gmm_diag3D = {
    'X': data.double(),
    'prior_counts': torch.ones(2).double(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': True
    }
}


gmm_full1F = {
    'X': data.float(),
    'prior_counts': torch.ones(10).float(),
    'comp_type': beer.NormalFullCovariance,
    'args': {
        'prior_mean': torch.zeros(2).float(),
        'prior_cov': torch.eye(2).float(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_full1D = {
    'X': data.double(),
    'prior_counts': torch.ones(10).double(),
    'comp_type': beer.NormalFullCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_full2F = {
    'X': data.float(),
    'prior_counts': torch.ones(1).float(),
    'comp_type': beer.NormalFullCovariance,
    'args': {
        'prior_mean': torch.zeros(2).float(),
        'prior_cov': torch.eye(2).float(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_full2D = {
    'X': data.double(),
    'prior_counts': torch.ones(1).double(),
    'comp_type': beer.NormalFullCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': False
    }
}

gmm_full3F = {
    'X': data.float(),
    'prior_counts': torch.ones(2).float(),
    'comp_type': beer.NormalFullCovariance,
    'args': {
        'prior_mean': torch.zeros(2).float(),
        'prior_cov': torch.eye(2).float(),
        'prior_count': 1,
        'random_init': True
    }
}

gmm_full3D = {
    'X': data.double(),
    'prior_counts': torch.ones(2).double(),
    'comp_type': beer.NormalFullCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': True
    }
}


tests = [
    (TestMixture, gmm_diag1F),
    (TestMixture, gmm_diag1D),
    (TestMixture, gmm_diag2F),
    (TestMixture, gmm_diag2D),
    (TestMixture, gmm_diag3F),
    (TestMixture, gmm_diag3D),
    (TestMixture, gmm_full1F),
    (TestMixture, gmm_full1D),
    (TestMixture, gmm_full2F),
    (TestMixture, gmm_full2D),
    (TestMixture, gmm_full3F),
    (TestMixture, gmm_full3D),
]

module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))


if __name__ == '__main__':
    unittest.main()

