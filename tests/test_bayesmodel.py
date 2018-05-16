'Test the BayesianModel module.'

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


def create_dirichlet(t_type):
    dim = int(1 + torch.randint(100, (1, 1)).item())
    concentrations = (torch.randn(dim) ** 2).type(t_type)
    return beer.DirichletPrior(concentrations)

def create_normalgamma(t_type):
    dim = int(1 + torch.randint(100, (1, 1)).item())
    mean = torch.randn(dim).type(t_type)
    scale = (1 + torch.randn(dim)**2).type(t_type)
    shape = (1 + torch.randn(dim)**2).type(t_type)
    rate = (1 + torch.randn(dim)**2).type(t_type)
    return beer.NormalGammaPrior(mean, scale, shape, rate)

def create_jointnormalgamma(t_type):
    ncomp = int(1 + torch.randint(100, (1, 1)).item())
    dim = int(1 + torch.randint(100, (1, 1)).item())
    means = torch.randn((ncomp, dim)).type(t_type)
    scales = (1 + torch.randn((ncomp, dim))**2).type(t_type)
    shape = (1 + torch.randn(dim)**2).type(t_type)
    rate = (1 + torch.randn(dim)**2).type(t_type)
    return beer.JointNormalGammaPrior(means, scales, shape, rate)

def create_normalwishart(t_type):
    dim = int(1 + torch.randint(100, (1, 1)).item())
    mean = torch.randn(dim).type(t_type)
    scale = (1 + torch.randn(1)**2).type(t_type)
    scale_mat = (1 + torch.randn(dim)).type(t_type)
    scale_mat = torch.eye(dim).type(t_type) + \
        torch.ger(scale_mat, scale_mat)
    dof = (dim + 100 + torch.randn(1)**2).type(t_type)
    return beer.NormalWishartPrior(mean, float(scale), scale_mat, float(dof))

def create_jointnormalwishart(t_type):
    ncomp = int(1 + torch.randint(100, (1, 1)).item())
    dim = int(1 + torch.randint(100, (1, 1)).item())
    means = torch.randn((ncomp, dim)).type(t_type)
    scales = (1 + torch.randn((ncomp))**2).type(t_type)
    scale_mat = (1 + torch.randn(dim)).type(t_type)
    scale_mat = torch.eye(dim).type(t_type) + \
        torch.ger(scale_mat, scale_mat)
    dof = (dim + 100 + torch.randn(1)**2).type(t_type)
    return beer.JointNormalWishartPrior(means, scales, scale_mat, float(dof))

def create_normalfull(t_type):
    dim = int(1 + torch.randint(100, (1, 1)).item())
    mean = torch.randn(dim).type(t_type)
    cov = (1 + torch.randn(dim)).type(t_type)
    cov = torch.eye(dim).type(t_type) + torch.ger(cov, cov)
    return beer.NormalFullCovariancePrior(mean, cov)

def create_normaliso(t_type):
    dim = int(1 + torch.randint(100, (1, 1)).item())
    mean = torch.randn(dim).type(t_type)
    var = (1 + torch.randn(dim)**2).type(t_type)
    return beer.NormalIsotropicCovariancePrior(mean, var)

def create_matrixnormal(t_type):
    dim1 = int(1 + torch.randint(100, (1, 1)).item())
    dim2 = int(1 + torch.randint(100, (1, 1)).item())
    mean = torch.randn(dim1, dim2).type(t_type)
    cov = (1 + torch.randn(dim1)).type(t_type)
    cov = torch.eye(dim1).type(t_type) + torch.ger(cov, cov)
    return beer.MatrixNormalPrior(mean, cov)

def create_gamma(t_type):
    shape = (1 + torch.randn(1) ** 2).type(t_type)
    rate = (1 + torch.randn(1) ** 2).type(t_type)
    return beer.GammaPrior(shape, rate)


class TestBayesianParameter(BaseTest):

    def setUp(self):
        self.priors = [
            create_dirichlet(self.type),
            create_normalgamma(self.type),
            create_jointnormalgamma(self.type),
            create_normalwishart(self.type),
            create_jointnormalwishart(self.type),
            create_normalfull(self.type),
            create_normaliso(self.type),
            create_matrixnormal(self.type),
            create_gamma(self.type),
        ]
        self.posteriors = self.priors


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
                    bayesparam.expected_value().numpy(),
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
                kl_div = beer.BayesianModel.kl_div_posterior_prior(
                    [beer.BayesianParameter(prior, posterior)
                     for _ in range(10)]
                )
        self.assertGreaterEqual(float(kl_div), 0.)


class TestBayesianParameterSet(BaseTest):

    def setUp(self):
        self.nparams = int(1 + torch.randint(20, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.concentrations = (torch.randn(self.dim) ** 2).type(self.type)
        self.shape = (1 + torch.randn(1) ** 2).type(self.type)
        self.rate = (1 + torch.randn(1) ** 2).type(self.type)
        self.priors, self.posteriors = [], []
        for _ in range(self.nparams):
            self.priors.append([
                create_dirichlet(self.type),
                create_normalgamma(self.type),
                create_jointnormalgamma(self.type),
                create_normalwishart(self.type),
                create_jointnormalwishart(self.type),
                create_normalfull(self.type),
                create_normaliso(self.type),
                create_matrixnormal(self.type),
                create_gamma(self.type),
            ])

            self.posteriors.append(self.priors[-1])

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
                        param.expected_value().numpy(),
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
