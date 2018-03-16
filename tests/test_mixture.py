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
        self.assertEqual(len(model.components), len(self.prior_counts))
        self.assertTrue(isinstance(model.components[0], self.comp_type))

    def test_sufficient_statistics(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        s1 = model.sufficient_statistics(self.X)
        s2 = self.comp_type.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1.numpy()[:, :-1], s2.numpy()))
        self.assertTrue(np.allclose(s1.numpy()[:, -1], np.ones(self.X.size(0))))

    def test_sufficient_statistics_from_mean_var(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        s1 = model.sufficient_statistics_from_mean_var(self.X,
            torch.ones_like(self.X))
        s2 = self.comp_type.sufficient_statistics_from_mean_var(self.X,
            torch.ones_like(self.X))
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

    def test_exp_llh_labels(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        labels = torch.zeros(self.X.size(0)).long()
        elabels = model._expand_labels(labels, len(model.components)).numpy()
        np_params_matrix = model._np_params_matrix.numpy()
        T = model.sufficient_statistics(self.X).numpy()
        per_component_exp_llh = T @ np_params_matrix.T
        exp_llh1 = logsumexp(per_component_exp_llh, axis=1)
        resps = elabels
        exp_llh1 -= .5 * self.X.shape[1] * math.log(2 * math.pi)
        acc_stats1 = resps.T @ T[:, :-1], resps.sum(axis=0)
        exp_llh2, acc_stats2 = model.exp_llh(self.X, accumulate=True,
            labels=labels)
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

    def test_split(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)

        prior_np = .5 * np.c_[model.prior_weights.natural_params.numpy(),
                              model.prior_weights.natural_params.numpy()].ravel()
        post_np = .5 * np.c_[model.posterior_weights.natural_params.numpy(),
                             model.posterior_weights.natural_params.numpy()].ravel()
        model2 = model.split()
        self.assertEqual(len(model2.components), 2 * len(model.components))
        self.assertTrue(np.allclose(model2.prior_weights.natural_params.numpy(),
            prior_np, atol=TOL))
        self.assertTrue(np.allclose(model2.posterior_weights.natural_params.numpy(),
            post_np, atol=TOL))

    def test_expected_natural_params(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)

        enp1, Ts1 = model.expected_natural_params(self.means, self.vars)
        T = model.components[0].sufficient_statistics_from_mean_var(self.means,
            self.vars).numpy()
        T2 = np.c_[T, np.ones(T.shape[0])]
        per_component_exp_llh = T2 @ model._np_params_matrix.numpy().T
        exp_llh = logsumexp(per_component_exp_llh, axis=1)
        resps = np.exp(per_component_exp_llh - exp_llh[:, None])
        matrix = np.c_[[component.expected_natural_params(self.means,
                        self.vars)[0][0].numpy()
                        for component in model.components]]
        acc_stats = resps.T @ T2[:, :-1], resps.sum(axis=0)
        enp2, Ts2 = resps @ matrix, acc_stats

        self.assertTrue(np.allclose(enp1.numpy(), enp2, atol=TOL))
        self.assertTrue(np.allclose(Ts1[0].numpy(), Ts2[0], atol=TOL))
        self.assertTrue(np.allclose(Ts1[1].numpy(), Ts2[1], atol=TOL))

    def test_expected_natural_params_labels(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        labels = torch.zeros(self.X.size(0)).long()
        elabels = model._expand_labels(labels, len(model.components)).numpy()
        enp1, Ts1 = model.expected_natural_params(self.means, self.vars,
            labels=labels)
        T = model.components[0].sufficient_statistics_from_mean_var(self.means,
            self.vars).numpy()
        T2 = np.c_[T, np.ones(T.shape[0])]
        per_component_exp_llh = T2 @ model._np_params_matrix.numpy().T
        exp_llh = logsumexp(per_component_exp_llh, axis=1)
        resps = elabels
        matrix = np.c_[[component.expected_natural_params(self.means,
                        self.vars)[0][0].numpy()
                        for component in model.components]]
        acc_stats = resps.T @ T2[:, :-1], resps.sum(axis=0)
        enp2, Ts2 = resps @ matrix, acc_stats

        self.assertTrue(np.allclose(enp1.numpy(), enp2, atol=TOL))
        self.assertTrue(np.allclose(Ts1[0].numpy(), Ts2[0], atol=TOL))
        self.assertTrue(np.allclose(Ts1[1].numpy(), Ts2[1], atol=TOL))

    def test_predictions(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        np_params_matrix = model._np_params_matrix.numpy()
        T = model.sufficient_statistics(self.X).numpy()
        per_component_exp_llh = T @ np_params_matrix.T
        exp_llh1 = logsumexp(per_component_exp_llh, axis=1)
        resps1 = np.exp(per_component_exp_llh - exp_llh1[:, None])
        resps2 = model.predictions(self.X)
        self.assertTrue(np.allclose(resps1, resps2.numpy(), atol=TOL))

    def test_predictions_from_mean_vars(self):
        model = beer.Mixture.create(self.prior_counts, self.comp_type.create,
            self.args)
        np_params_matrix = model._np_params_matrix.numpy()
        T = model.sufficient_statistics_from_mean_var(self.X,
            torch.ones_like(self.X)).numpy()
        per_component_exp_llh = T @ np_params_matrix.T
        exp_llh1 = logsumexp(per_component_exp_llh, axis=1)
        resps1 = np.exp(per_component_exp_llh - exp_llh1[:, None])
        resps2 = model.predictions_from_mean_var(self.X, torch.ones_like(self.X))
        self.assertTrue(np.allclose(resps1, resps2.numpy(), atol=TOL))

    def test_expand_labels(self):
        ref = torch.range(0, 2).long()
        labs1 = beer.Mixture._expand_labels(ref, 3).long()
        labs2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(labs1.numpy(), labs2, atol=TOL))


torch.manual_seed(10)
dataF = {
    'X': torch.randn(20, 2).float(),
    'means': torch.randn(20, 2).float(),
    'vars': torch.randn(20, 2).float() ** 2
}

dataD = {
    'X': torch.randn(20, 2).double(),
    'means': torch.randn(20, 2).double(),
    'vars': torch.randn(20, 2).double() ** 2
}

gmm_diag1F = {
    **dataF,
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
    **dataD,
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
    **dataF,
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
    **dataD,
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
    **dataF,
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
    **dataD,
    'prior_counts': torch.ones(2).double(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': True
    }
}

gmm_diag4F = {
    **dataF,
    'prior_counts': torch.range(1, 10).float(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).float(),
        'prior_cov': torch.eye(2).float(),
        'prior_count': 1,
        'random_init': True
    }
}

gmm_diag4D = {
    **dataD,
    'prior_counts': torch.range(1, 10).double(),
    'comp_type': beer.NormalDiagonalCovariance,
    'args': {
        'prior_mean': torch.zeros(2).double(),
        'prior_cov': torch.eye(2).double(),
        'prior_count': 1,
        'random_init': True
    }
}


gmm_full1F = {
    **dataF,
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
    **dataD,
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
    **dataF,
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
    **dataD,
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
    **dataF,
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
    **dataD,
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
    (TestMixture, gmm_diag4F),
    (TestMixture, gmm_diag4D),
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

