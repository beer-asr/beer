'Test the Normal model.'



import unittest
import numpy as np
import math
import torch

import sys
sys.path.insert(0, './')

import beer

torch.manual_seed(10)


TOLPLACES = 5
TOL = 10 ** (-TOLPLACES)


class TestSVBLoss:

    def test_loss(self):
        loss_fn = beer.StochasticVariationalBayesLoss(len(self.X))
        loss1 = loss_fn(self.model, self.X, self.labels)
        T = self.model.sufficient_statistics(self.X)
        loss2 = self.model(T, self.labels).sum() - \
            beer.kl_div_posterior_prior(self.model.parameters)
        loss2 = torch.sum(loss2)
        self.assertAlmostEqual(float(loss1), float(loss2))
        self.assertAlmostEqual(float(loss1), float(loss1.exp_llh - loss1.kl_div))
        self.assertEqual(len(loss1.exp_llh_per_frame), len(self.X))
        self.assertAlmostEqual(float(loss1.exp_llh), float(loss1.exp_llh.sum()))
        self.assertAlmostEqual(float(loss1.value),
            float(loss1.exp_llh.sum() - loss1.kl_div))

    def test_loss_batch(self):
        bsize = 10
        if self.labels is not None:
            labels = self.labels[:bsize]
        else:
            labels = None
        scale = float(bsize) / float(len(self.X))
        loss_fn = beer.StochasticVariationalBayesLoss(len(self.X))
        loss1 = loss_fn(self.model, self.X[:bsize], labels)
        T = self.model.sufficient_statistics(self.X[:bsize])
        loss2 = scale * self.model(T, labels).sum() - \
            beer.kl_div_posterior_prior(self.model.parameters)
        loss2 = torch.sum(loss2)
        self.assertAlmostEqual(float(loss1), float(loss2))
        self.assertAlmostEqual(float(loss1),
            float(scale * loss1.exp_llh - loss1.kl_div))
        self.assertEqual(len(loss1.exp_llh_per_frame), bsize)
        self.assertAlmostEqual(float(loss1.exp_llh),
            float(loss1.exp_llh.sum()))
        self.assertAlmostEqual(float(loss1.value),
            float(scale * loss1.exp_llh.sum() - loss1.kl_div))


class TestBayesianModelOptimizer:

    def test_optimizer(self):
        loss_fn = beer.StochasticVariationalBayesLoss(len(self.X))
        optimizer = beer.BayesianModelOptimizer(self.model.parameters,
            lrate=self.lrate)

        previous_loss = -1e5
        for i in range(10):
            loss = loss_fn(self.model, self.X)

            optimizer.zero_natural_grad()
            loss.backward_natural_grad()
            optimizer.step()

            self.assertGreaterEqual(float(loss), float(previous_loss))
            previous_loss = loss


dir1_prior = beer.DirichletPrior(torch.ones(2))
dir2_prior = beer.DirichletPrior(torch.ones(2) + 1.)
ng1_prior = beer.NormalGammaPrior(torch.zeros(2), torch.ones(2), 1.)
ng2_prior = beer.NormalGammaPrior(torch.ones(2), torch.ones(2), 1.)
nw1_prior = beer.NormalWishartPrior(torch.zeros(2), torch.eye(2), 1.)
nw2_prior = beer.NormalWishartPrior(torch.ones(2), torch.eye(2), 1.)

n1_model = beer.NormalDiagonalCovariance(ng1_prior, ng2_prior)
n2_model = beer.NormalFullCovariance(nw2_prior, nw2_prior)


tests = [
    (TestSVBLoss, {'model': n1_model, 'X': torch.randn(20, 2).float(),
        'labels': None}),
    (TestSVBLoss, {'model': n2_model, 'X': torch.randn(20, 2).float(),
        'labels': None}),
    (TestSVBLoss, {'model': n1_model, 'X': torch.randn(20, 2).float(),
        'labels': torch.ones(20).long()}),
    (TestSVBLoss, {'model': n2_model, 'X': torch.randn(20, 2).float(),
        'labels': torch.ones(20).long()}),

    (TestBayesianModelOptimizer, {'model': n1_model,
        'X': torch.randn(20, 2).float(), 'labels': None, 'lrate': 1.}),
    (TestBayesianModelOptimizer, {'model': n1_model,
        'X': torch.randn(20, 2).float(), 'labels': None, 'lrate': 1.}),
    (TestBayesianModelOptimizer, {'model': n1_model,
        'X': torch.randn(20, 2).float(), 'labels': torch.ones(20),
        'lrate': 1.}),
    (TestBayesianModelOptimizer, {'model': n1_model,
        'X': torch.randn(20, 2).float(), 'labels': torch.ones(20),
        'lrate': 1.}),

]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

