'Test the Normal model.'



import unittest
import numpy as np
import math
import torch

import sys
sys.path.insert(0, './')
import beer
from beer import NormalDiagonalCovariance


torch.manual_seed(10)


TOLPLACES = 5
TOL = 10 ** (-TOLPLACES)


class TestNormalDiagonalCovariance:

    def test_create(self):
        model = beer.NormalDiagonalCovariance.create(self.mean, self.cov,
            prior_count=self.prior_count)
        m1, m2 = self.mean.numpy(), model.mean.numpy()
        self.assertTrue(np.allclose(m1, m2, atol=TOL))
        c1, c2 = self.cov.numpy(), model.cov.numpy()
        if len(c1.shape) == 1:
            c1 = np.diag(c1)
        self.assertTrue(np.allclose(c1, c2, atol=TOL))
        self.assertAlmostEqual(self.prior_count, float(model.count),
                               places=TOLPLACES)

    def test_create_random(self):
        model = beer.NormalDiagonalCovariance.create(self.mean, self.cov,
            prior_count=self.prior_count, random_init=True)
        m1, m2 = self.mean.numpy(), model.mean.numpy()
        self.assertFalse(np.allclose(m1, m2))
        c1, c2 = self.cov.numpy(), model.cov.numpy()
        if len(c1.shape) == 1:
            c1 = np.diag(c1)
        self.assertTrue(np.allclose(c1, c2, atol=TOL))
        self.assertAlmostEqual(self.prior_count, float(model.count),
            places=TOLPLACES)

    def test_sufficient_statistics(self):
        X =  self.X.numpy()
        s1 = np.c_[self.X**2, self.X, np.ones_like(X), np.ones_like(X)]
        s2 = beer.NormalDiagonalCovariance.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1, s2.numpy(), atol=TOL))

    def test_sufficient_statistics_from_mean_var(self):
        mean = self.mean.view(1, -1)
        var = self.cov.view(1, -1)
        if len(var.size()) == 2:
            var = torch.diag(var)
        s1 = beer.NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            mean, var)
        mean, var = mean.numpy(), var.numpy()
        s2 = np.c_[mean**2 + var, mean, np.ones_like(mean),
                   np.ones_like(mean)]
        self.assertTrue(np.allclose(s1.numpy(), s2, atol=TOL))

    def test_exp_llh(self):
        model = beer.NormalDiagonalCovariance.create(self.mean, self.cov,
            self.prior_count)
        T = model.sufficient_statistics(self.X)
        nparams = model.posterior.expected_sufficient_statistics
        exp_llh1 = T @ nparams
        exp_llh1 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        s1 = T.sum(dim=0)
        exp_llh2, s2 = model.exp_llh(self.X, accumulate=True)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy(),
                        atol=TOL))
        self.assertTrue(np.allclose(s1.numpy(), s2.numpy(), rtol=TOL,
                        atol=TOL))

    def test_split(self):
        model = beer.NormalDiagonalCovariance.create(self.mean, self.cov,
            self.prior_count)
        smodel1, smodel2 = model.split()
        cov = model.cov.numpy()
        evals, evecs = np.linalg.eigh(cov)
        m1 = model.mean.numpy() + evecs.T @ np.sqrt(evals)
        m2 = model.mean.numpy() - evecs.T @ np.sqrt(evals)
        self.assertTrue(np.allclose(m1, smodel1.mean.numpy(), atol=TOL))
        self.assertTrue(np.allclose(m2, smodel2.mean.numpy(), atol=TOL))

    def test_expected_natural_params(self):
        model = beer.NormalDiagonalCovariance.create(self.mean, self.cov,
            self.prior_count)
        mean = self.mean.view(1, -1)
        if len(self.cov.size()) == 2:
            var = torch.diag(self.cov)
        else:
            var = self.cov
        var = var.view(1, -1)
        enp1, Ts1 = model.expected_natural_params(mean, var)
        T = model.sufficient_statistics_from_mean_var(mean, var).numpy()
        np1, np2, np3, np4 = \
            model.posterior.expected_sufficient_statistics.view(4, -1).numpy()
        identity = np.eye(var.shape[1])
        np1 = (np1[:, None] * identity[None, :, :]).reshape(-1)
        enp2, Ts2 = np.c_[np1[None], np2[None], np3.sum(axis=-1)[None],
                    np4.sum(axis=-1)[None]], T.sum(axis=0)
        self.assertTrue(np.allclose(enp1.numpy(), enp2, atol=TOL))
        self.assertTrue(np.allclose(Ts1.numpy(), Ts2, atol=TOL))



class TestNormalFullCovariance:

    def test_create(self):
        model = beer.NormalFullCovariance.create(self.mean, self.cov,
            self.prior_count)
        m1, m2 = self.mean.numpy(), model.mean.numpy()
        self.assertTrue(np.allclose(m1, m2))
        c1, c2 = self.cov.numpy(), model.cov.numpy()
        self.assertTrue(np.allclose(c1, c2, atol=TOL))
        self.assertAlmostEqual(self.prior_count, model.count, places=TOLPLACES)

    def test_create_random(self):
        model = beer.NormalFullCovariance.create(self.mean, self.cov,
            self.prior_count, random_init=True)
        m1, m2 = self.mean.numpy(), model.mean.numpy()
        self.assertFalse(np.allclose(m1, m2))
        c1, c2 = self.cov.numpy(), model.cov.numpy()
        self.assertTrue(np.allclose(c1, c2, atol=TOL))
        self.assertAlmostEqual(self.prior_count, model.count, places=TOLPLACES)

    def test_sufficient_statistics(self):
        X = self.X.numpy()
        s1 = np.c_[(X[:, :, None] * X[:, None, :]).reshape(len(X), -1),
            X, np.ones(len(X)), np.ones(len(X))]
        s2 = beer.NormalFullCovariance.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1, s2.numpy(), atol=TOL))

    def test_sufficient_statistics_from_mean_var(self):
        mean = self.mean.view(1, -1)
        var = torch.diag(self.cov).view(1, -1)
        s1 = beer.NormalFullCovariance.sufficient_statistics_from_mean_var(
            mean, var)
        mean, var = mean.numpy(), var.numpy()
        idxs = np.identity(mean.shape[1]).reshape(-1) == 1
        XX = (mean[:, :, None] * mean[:, None, :]).reshape(mean.shape[0], -1)
        XX[:, idxs] += var
        s2 = np.c_[XX, mean, np.ones(len(mean)), np.ones(len(mean))]
        self.assertTrue(np.allclose(s1.numpy(), s2, atol=TOL))

    def test_exp_llh(self):
        model = beer.NormalFullCovariance.create(self.mean, self.cov,
            self.prior_count)
        T = model.sufficient_statistics(self.X)
        nparams = model.posterior.expected_sufficient_statistics
        exp_llh1 = T @ nparams
        exp_llh1 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        s1 = T.sum(dim=0)
        exp_llh2, s2 = model.exp_llh(self.X, accumulate=True)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy(),
                        atol=TOL))
        self.assertTrue(np.allclose(s1.numpy(), s2.numpy(), atol=TOL))

    def test_split(self):
        model = beer.NormalFullCovariance.create(self.mean, self.cov,
            self.prior_count)
        smodel1, smodel2 = model.split()

        cov = model.cov.numpy()
        evals, evecs = np.linalg.eigh(cov)
        m1 = model.mean.numpy() + evecs.T @ np.sqrt(evals)
        m2 = model.mean.numpy() - evecs.T @ np.sqrt(evals)
        self.assertTrue(np.allclose(m1, smodel1.mean.numpy(), atol=TOL))
        self.assertTrue(np.allclose(m2, smodel2.mean.numpy(), atol=TOL))

    def test_expected_natural_params(self):
        model = beer.NormalFullCovariance.create(self.mean, self.cov,
            self.prior_count)
        mean = self.mean.view(1, -1)
        var = torch.diag(self.cov).view(1, -1)
        enp1, Ts1 = model.expected_natural_params(mean, var)
        T = model.sufficient_statistics_from_mean_var(mean, var).numpy()
        enp2, Ts2 = model.posterior.expected_sufficient_statistics.numpy(), \
             T.sum(axis=0)
        self.assertTrue(np.allclose(enp1.numpy(), enp2, atol=TOL))
        self.assertTrue(np.allclose(Ts1.numpy(), Ts2, atol=TOL))


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

data10F = {
    'X': torch.randn(20, 10).float(),
    'means': torch.randn(20, 10).float(),
    'vars': torch.randn(20, 10).float() ** 2
}

data10D = {
    'X': torch.randn(20, 10).double(),
    'means': torch.randn(20, 10).double(),
    'vars': torch.randn(20, 2).double() ** 2
}


tests = [
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float(), 'prior_count': 1., **dataF}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.ones(2).double(), 'prior_count': 1., **dataD}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(10).float(), 'cov': torch.ones(10).float(), 'prior_count': 1., **data10F}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(10).double(), 'cov': torch.ones(10).double(), 'prior_count': 1., **data10D}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float(), 'prior_count': 1., **dataF}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double(), 'prior_count': 1., **dataD}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float(), 'prior_count': 1e-3, **dataF}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.ones(2).double(), 'prior_count': 1e-8, **dataD}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float() * 1e-5, 'prior_count': 1., **dataF}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.ones(2).double() * 1e-8, 'prior_count': 1., **dataD}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float() * 1e2, 'prior_count': 1., **dataF}),

    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float(), 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double(), 'prior_count': 1., **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.FloatTensor([[2, -1.2], [-1.2, 10.]]).float(), 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.DoubleTensor([[2, -1.2], [-1.2, 10.]]).float(), 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(10).float(), 'cov': torch.eye(10).float(), 'prior_count': 1., **data10F}),
    (TestNormalFullCovariance, {'mean': torch.ones(10).double(), 'cov': torch.eye(10).double(), 'prior_count': 1., **data10D}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float(), 'prior_count': 1e-3, **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double(), 'prior_count': 1e-7, **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float() * 1e-5, 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double() * 1e-8, 'prior_count': 1., **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float() * 1e2, 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., **dataD}),
]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

