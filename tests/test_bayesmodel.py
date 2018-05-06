'Test the Normal model.'



import unittest
import numpy as np
import math
import torch

import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import beer
from basetest import BaseTest

torch.manual_seed(10)


TOLPLACES = 5
TOL = 10 ** (-TOLPLACES)


class TestBayesianParameter:

    def test_create(self):
        bayesparam = beer.BayesianParameter(self.prior, self.posterior)
        self.assertArraysAlmostEqual(self, bayesparam.natura_grad.numpy(),
            np.zeros(len(bayesparam.natural_grad)))

    def test_expected_value(self):
        bayesparam = beer.BayesianParameter(self.prior, self.posterior)
        self.assertTrue(np.allclose(bayesparam.expected_value.numpy(),
            self.posterior.expected_sufficient_statistics.numpy()))

    def test_zero_natural_grad(self):
        bayesparam = beer.BayesianParameter(self.prior, self.posterior)
        bayesparam.natural_grad += 1
        bayesparam.zero_natural_grad()
        self.assertTrue(np.allclose(bayesparam.natural_grad.numpy(),
            np.zeros(len(bayesparam.natural_grad))))

    def test_kl_div_prior_posterior(self):
        kl_div = float(beer.kl_div_posterior_prior([
            beer.BayesianParameter(self.prior, self.posterior) for _ in range(10)
        ]))
        self.assertAlmostEqual(kl_div, 0.)


class TestBayesianParameterSet:

    def test_create(self):
        params = [beer.BayesianParameter(self.prior, self.posterior)
            for _ in range(self.nparams)]
        param_set = beer.BayesianParameterSet(params)
        self.assertEqual(len(params), len(param_set))


dir_prior = beer.DirichletPrior(torch.ones(22))
ng_prior = beer.NormalGammaPrior(torch.zeros(2), torch.ones(2), 1.)
nw_prior = beer.NormalWishartPrior(torch.zeros(2), torch.eye(2), 1.)


tests = [
    (TestBayesianParameter, {'prior': dir_prior, 'posterior': dir_prior}),
    (TestBayesianParameter, {'prior': ng_prior, 'posterior': ng_prior}),
    (TestBayesianParameter, {'prior': nw_prior, 'posterior': nw_prior}),
]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (BaseTest, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

