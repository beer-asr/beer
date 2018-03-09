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


data = torch.randn(20, 2)
data10 = torch.randn(20, 10)
tests = [
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float(), 'prior_count': 1., 'X': data.float()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.ones(2).double(), 'prior_count': 1., 'X': data.double()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(10).float(), 'cov': torch.ones(10).float(), 'prior_count': 1., 'X': data10.float()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(10).double(), 'cov': torch.ones(10).double(), 'prior_count': 1., 'X': data10.double()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float(), 'prior_count': 1., 'X': data.float()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double(), 'prior_count': 1., 'X': data.double()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float(), 'prior_count': 1e-3, 'X': data.float()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.ones(2).double(), 'prior_count': 1e-8, 'X': data.double()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float() * 1e-5, 'prior_count': 1., 'X': data.float()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'cov': torch.ones(2).double() * 1e-8, 'prior_count': 1., 'X': data.double()}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'cov': torch.ones(2).float() * 1e2, 'prior_count': 1., 'X': data.float()}),

    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float(), 'prior_count': 1., 'X': data.float()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double(), 'prior_count': 1., 'X': data.double()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.FloatTensor([[2, -1.2], [-1.2, 10.]]).float(), 'prior_count': 1., 'X': data.float()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.DoubleTensor([[2, -1.2], [-1.2, 10.]]).float(), 'prior_count': 1., 'X': data.float()}),
    (TestNormalFullCovariance, {'mean': torch.ones(10).float(), 'cov': torch.eye(10).float(), 'prior_count': 1., 'X': data10.float()}),
    (TestNormalFullCovariance, {'mean': torch.ones(10).double(), 'cov': torch.eye(10).double(), 'prior_count': 1., 'X': data10.double()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float(), 'prior_count': 1e-3, 'X': data.float()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double(), 'prior_count': 1e-7, 'X': data.double()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float() * 1e-5, 'prior_count': 1., 'X': data.float()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double() * 1e-8, 'prior_count': 1., 'X': data.double()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(), 'cov': torch.eye(2).float() * 1e2, 'prior_count': 1., 'X': data.float()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., 'X': data.double()}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(), 'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., 'X': data.double()}),
]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

