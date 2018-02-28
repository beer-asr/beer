'Test the Normal model.'


import sys
sys.path.insert(0, './')
import unittest
import numpy as np
import beer
from beer import NormalDiagonalCovariance
import torch


class TestNormalDiagonalCovariance(unittest.TestCase):

    def test_create(self):
        mean, prec = torch.ones(4), 2 * torch.eye(4)
        model = beer.NormalDiagonalCovariance.create(mean, prec,
            prior_count=2.5)
        m1, m2 = mean.numpy(), model.mean.numpy()
        self.assertTrue(np.allclose(m1, m2))
        c1, c2 = prec.numpy(), model.cov.numpy()
        self.assertTrue(np.allclose(c1, c2))
        self.assertAlmostEqual(2.5, model.count)

    def test_create_random(self):
        mean, prec = torch.ones(4), 2 * torch.eye(4)
        model = beer.NormalDiagonalCovariance.create(mean, prec,
            prior_count=2.5, random_init=True)
        m1, m2 = mean.numpy(), model.mean.numpy()
        self.assertFalse(np.allclose(m1, m2))
        c1, c2 = prec.numpy(), model.cov.numpy()
        self.assertTrue(np.allclose(c1, c2))
        self.assertAlmostEqual(2.5, model.count)

    def test_sufficient_statistics(self):
        X = np.ones((20, 5)) * 3.14
        s1 = np.c_[X**2, X, np.ones_like(X), np.ones_like(X)]
        X = torch.from_numpy(X)
        s2 = beer.NormalDiagonalCovariance.sufficient_statistics(X)
        self.assertTrue(np.allclose(s1, s2.numpy()))



if __name__ == '__main__':
    unittest.main()
