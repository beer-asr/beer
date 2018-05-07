'Test the Normal model.'

# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import numpy as np
import torch
import beer
from basetest import BaseTest


class TestBayesianParameter(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.priors = [
            beer.DirichletPrior(torch.ones(self.dim).type(self.type)),
            beer.NormalGammaPrior(torch.randn(self.dim).type(self.type),
                                  (torch.randn(self.dim)**2).type(self.type),
                                  self.prior_count)
        ]

        self.posteriors = [
            beer.DirichletPrior(torch.ones(self.dim).type(self.type)),
            beer.NormalGammaPrior(torch.randn(self.dim).type(self.type),
                                  (torch.randn(self.dim)**2).type(self.type),
                                  self.prior_count)
        ]
        #self.ng_prior = beer.NormalGammaPrior(torch.zeros(2), torch.ones(2), 1.)
        #nw_prior = beer.NormalWishartPrior(torch.zeros(2), torch.eye(2), 1.)

    def test_create(self):
        for i, pdfs in enumerate(zip(self.priors, self.posteriors)):
            prior, posterior = pdfs
            with self.subTest(i=i):
                bayesparam = beer.BayesianParameter(prior, posterior)
                self.assertArraysAlmostEqual(
                    bayesparam.natural_grad.numpy(),
                    np.zeros(len(bayesparam.natural_grad))
                )

    def test_expected_value(self):
        for i, pdfs in enumerate(zip(self.priors, self.posteriors)):
            prior, posterior = pdfs
            with self.subTest(i=i):
                bayesparam = beer.BayesianParameter(prior, posterior)
                self.assertArraysAlmostEqual(
                    bayesparam.expected_value.numpy(),
                    posterior.expected_sufficient_statistics.numpy()
                )

    def test_zero_natural_grad(self):
        for i, pdfs in enumerate(zip(self.priors, self.posteriors)):
            prior, posterior = pdfs
            with self.subTest(i=i):
                bayesparam = beer.BayesianParameter(prior, posterior)
                bayesparam.natural_grad += 1
                bayesparam.zero_natural_grad()
                self.assertArraysAlmostEqual(
                    bayesparam.natural_grad.numpy(),
                    np.zeros(len(bayesparam.natural_grad))
                )

    def test_kl_div_prior_posterior(self):
        for i, pdfs in enumerate(zip(self.priors, self.posteriors)):
            prior, posterior = pdfs
            with self.subTest(i=i):
                kl_div = beer.kl_div_posterior_prior(
                    [beer.BayesianParameter(prior, posterior)
                     for _ in range(10)]
                )
        self.assertGreaterEqual(float(kl_div), 0.)


class TestBayesianParameterSet(BaseTest):

    def setUp(self):
        self.nparams = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.priors, self.posteriors = [], []
        for _ in range(self.nparams):
            self.priors.append([
                beer.DirichletPrior(torch.ones(self.dim).type(self.type)),
                beer.NormalGammaPrior(torch.randn(self.dim).type(self.type),
                                      (torch.randn(self.dim)**2).type(self.type),
                                      self.prior_count)
            ])

            self.posteriors.append([
                beer.DirichletPrior(torch.ones(self.dim).type(self.type)),
                beer.NormalGammaPrior(torch.randn(self.dim).type(self.type),
                                      (torch.randn(self.dim)**2).type(self.type),
                                      self.prior_count)
            ])

    def test_create(self):
        for i in range(self.nparams):
            with self.subTest(i=i):
                pdfs = zip(self.priors[i], self.posteriors[i])
                params = [beer.BayesianParameter(prior, posterior)
                          for prior, posterior in pdfs]
                param_set = beer.BayesianParameterSet(params)
                self.assertEqual(len(params), len(param_set))
                for param in param_set:
                    self.assertArraysAlmostEqual(
                        param.natural_grad.numpy(),
                        np.zeros(len(param.natural_grad))
                    )

    def test_expected_value(self):
        for i in range(self.nparams):
            with self.subTest(i=i):
                pdfs = zip(self.priors[i], self.posteriors[i])
                params = [beer.BayesianParameter(prior, posterior)
                          for prior, posterior in pdfs]
                param_set = beer.BayesianParameterSet(params)
                for j, param in enumerate(param_set):
                    posterior = self.posteriors[i][j]
                    self.assertArraysAlmostEqual(
                        param.expected_value.numpy(),
                        posterior.expected_sufficient_statistics.numpy()
                    )

    def test_zero_natural_grad(self):
        for i in range(self.nparams):
            with self.subTest(i=i):
                pdfs = zip(self.priors[i], self.posteriors[i])
                params = [beer.BayesianParameter(prior, posterior)
                          for prior, posterior in pdfs]
                param_set = beer.BayesianParameterSet(params)
                for param in param_set:
                    param.natural_grad += 1
                    param.zero_natural_grad()
                    self.assertArraysAlmostEqual(
                        param.natural_grad.numpy(),
                        np.zeros(len(param.natural_grad))
                    )


__all__ = ['TestBayesianParameter', 'TestBayesianParameterSet']
