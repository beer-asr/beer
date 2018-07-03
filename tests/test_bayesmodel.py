'Test the BayesianModel module.'

# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')
import glob
import yaml
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


class TestBayesianModel(BaseTest):

    def setUp(self):
        self.dim = int(10 + torch.randint(100, (1, 1)).item())
        self.dim = 2

        self.conf_files = []
        for path in glob.glob('./tests/models/*yml'):
            self.conf_files.append(path)
        assert len(self.conf_files) > 0

        self.mean = torch.randn(self.dim).type(self.type)
        self.variance = (1 + torch.randn(self.dim) ** 2).type(self.type)
        self.models1 = []
        for conf_file in self.conf_files:
            with open(conf_file, 'r') as fid:
                conf = yaml.load(fid)
            model = beer.create_model(conf, self.mean, self.variance)
            self.models1.append(model)

        self.mean = torch.randn(self.dim).type(self.type)
        self.variance = (1 + torch.randn(self.dim) ** 2).type(self.type)
        self.models2 = []
        for conf_file in self.conf_files:
            with open(conf_file, 'r') as fid:
                conf = yaml.load(fid)
            model = beer.create_model(conf, self.mean, self.variance)
            self.models2.append(model)

        self.weights = (1 + torch.randn(2)**2).type(self.type)
        self.weights /= self.weights.sum()

    def test_average_models(self):
        for i, model1, model2 in zip(range(len(self.models1)), self.models1,
                            self.models2):
            with self.subTest(model=model1):
                new_model = beer.average_models([model1, model2], self.weights)
                for param1, param2, new_param in zip(model1.parameters,
                                                     model2.parameters,
                                                     new_model.parameters):
                    p1 = param1.prior.natural_hparams.numpy()
                    nparam = new_param.prior.natural_hparams.numpy()
                    self.assertArraysAlmostEqual(p1, nparam)

                    p1 = param1.posterior.natural_hparams.numpy()
                    p2 = param2.posterior.natural_hparams.numpy()
                    np = new_param.posterior.natural_hparams.numpy()
                    w1, w2 = self.weights.numpy()
                    self.assertArraysAlmostEqual(w1 * p1 + w2 * p2, np)

                for param1, param2, new_param in zip(model1.non_bayesian_parameters(),
                                                     model2.non_bayesian_parameters(),
                                                     new_model.non_bayesian_parameters()):
                    w1, w2 = self.weights.numpy()
                    p1 = param1.data.numpy()
                    p2 = param2.data.numpy()
                    avg_p = w1 * p1 + w2 * p2
                    nparam = new_param.data.numpy()
                    self.assertArraysAlmostEqual(avg_p, nparam)


__all__ = [
    'TestBayesianParameter',
    'TestBayesianParameterSet',
    'TestBayesianModel'
]
