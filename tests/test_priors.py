'Test the priors package.'

import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import unittest
import torch
import beer
from basetest import BaseTest


class BaseTestPrior(BaseTest):

    def test_exp_sufficient_statistics(self):
        stats1 = self.prior.expected_sufficient_statistics()
        copied_tensor = torch.tensor(self.prior.natural_parameters,
                                     requires_grad=True)
        log_norm = self.prior.log_norm(copied_tensor)
        torch.autograd.backward(log_norm)
        stats2 = copied_tensor.grad
        self.assertArraysAlmostEqual(stats1.numpy(), stats2.numpy())

########################################################################
# Dirichlet.
########################################################################

class TestDirichletPrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        self.std_parameters = 2 * torch.ones(dim)
        self.prior = beer.DirichletPrior(self.std_parameters)

    def test_natural2std(self):
        std_params = self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(
            std_params.numpy(),
            self.std_parameters.numpy()
        )

    def test_std2natural(self):
        std_params = self.prior.to_std_parameters(self.prior.natural_parameters)
        nparams = self.prior.to_natural_parameters(std_params)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


########################################################################
# Gamma.
########################################################################

class TestGammaPrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        self.shape = torch.tensor(2).type(self.type)
        self.rate = torch.tensor(.5).type(self.type)
        self.prior = beer.GammaPrior(self.shape, self.rate)

    def test_natural2std(self):
        shape, rate = self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(shape.numpy(), self.shape.numpy())
        self.assertArraysAlmostEqual(rate.numpy(), self.rate.numpy())

    def test_std2natural(self):
        shape, rate = self.prior.to_std_parameters(self.prior.natural_parameters)
        nparams = self.prior.to_natural_parameters(shape, rate)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


########################################################################
# Wishart.
########################################################################

class TestWishartPrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        self.scale = torch.eye(dim).type(self.type)
        self.dof = torch.tensor(dim + 2).type(self.type)
        self.prior = beer.WishartPrior(self.scale, self.dof)

    def test_natural2std(self):
        scale, dof = self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(scale.numpy(), self.scale.numpy())
        self.assertArraysAlmostEqual(dof.numpy(), self.dof.numpy())

    def test_std2natural(self):
        scale, dof = self.prior.to_std_parameters(self.prior.natural_parameters)
        nparams = self.prior.to_natural_parameters(scale, dof)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


########################################################################
# Normal Full covariance.
########################################################################

class TestNormalFullCovariancePrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        self.scale = torch.eye(dim).type(self.type)
        self.dof = torch.tensor(dim + 2).type(self.type)
        self.prior_precision = beer.WishartPrior(self.scale, self.dof)
        self.mean = 3 * torch.ones(dim).type(self.type)
        self.scale = torch.tensor(1.5).type(self.type)
        self.prior = beer.NormalFullCovariancePrior(self.mean, self.scale,
                                                    self.prior_precision)

    def test_natural2std(self):
        mean, scale = self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(mean.numpy(), self.mean.numpy())
        self.assertArraysAlmostEqual(scale.numpy(), self.scale.numpy())

    def test_std2natural(self):
        mean, scale = self.prior.to_std_parameters(self.prior.natural_parameters)
        nparams = self.prior.to_natural_parameters(mean, scale)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


########################################################################
# Normal Wishart.
########################################################################

class TestNormalWishartPrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        self.mean = 3 + torch.zeros(dim).type(self.type)
        self.scale = torch.tensor(2.5).type(self.type)
        self.mean_precision = torch.eye(dim).type(self.type)
        self.dof = torch.tensor(dim + 2).type(self.type)
        self.prior = beer.NormalWishartPrior(self.mean, self.scale,
                                             self.mean_precision, self.dof)

    def test_natural2std(self):
        mean, scale, mean_precision, dof = \
            self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(mean.numpy(), self.mean.numpy())
        self.assertArraysAlmostEqual(scale.numpy(), self.scale.numpy())
        self.assertArraysAlmostEqual(mean_precision.numpy(), self.mean_precision.numpy())
        self.assertArraysAlmostEqual(dof.numpy(), self.dof.numpy())

    def test_std2natural(self):
        mean, scale, mean_precision, dof = self.prior.to_std_parameters()
        nparams = self.prior.to_natural_parameters(mean, scale, mean_precision, dof)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


########################################################################
# Normal Gamma.
########################################################################

class TestNormalGammaPrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        self.mean = 3 + torch.zeros(dim).type(self.type)
        self.scale = torch.tensor(2.5).type(self.type)
        self.shape = torch.tensor(3).type(self.type)
        self.rates = 2* torch.ones(dim).type(self.type)
        self.prior = beer.NormalGammaPrior(self.mean, self.scale,
                                           self.shape, self.rates)

    def test_natural2std(self):
        mean, scale, shape, rates = \
            self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(mean.numpy(), self.mean.numpy())
        self.assertArraysAlmostEqual(scale.numpy(), self.scale.numpy())
        self.assertArraysAlmostEqual(shape.numpy(), self.shape.numpy())
        self.assertArraysAlmostEqual(rates.numpy(), self.rates.numpy())

    def test_std2natural(self):
        mean, scale, shape, rates = self.prior.to_std_parameters()
        nparams = self.prior.to_natural_parameters(mean, scale, shape, rates)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


########################################################################
# Isotropic Normal Gamma.
########################################################################

class TestIsotropicNormalGammaPrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        self.mean = 3 + torch.zeros(dim).type(self.type)
        self.scale = torch.tensor(2.5).type(self.type)
        self.shape = torch.tensor(3).type(self.type)
        self.rate = torch.tensor(2).type(self.type)
        self.prior = beer.IsotropicNormalGammaPrior(self.mean, self.scale,
                                                    self.shape, self.rate)

    def test_natural2std(self):
        mean, scale, shape, rate = \
            self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(mean.numpy(), self.mean.numpy())
        self.assertArraysAlmostEqual(scale.numpy(), self.scale.numpy())
        self.assertArraysAlmostEqual(shape.numpy(), self.shape.numpy())
        self.assertArraysAlmostEqual(rate.numpy(), self.rate.numpy())

    def test_std2natural(self):
        mean, scale, shape, rate = self.prior.to_std_parameters()
        nparams = self.prior.to_natural_parameters(mean, scale, shape, rate)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


########################################################################
# Jiont Isotropic Normal Gamma.
########################################################################

class TestJointIsotropicNormalGammaPrior(BaseTestPrior):

    def setUp(self):
        dim = 10
        k = 3
        self.means = 3 + torch.zeros(k, dim).type(self.type)
        self.scales = 2.5 * torch.ones(k).type(self.type)
        self.shape = torch.tensor(3).type(self.type)
        self.rate = torch.tensor(2).type(self.type)
        self.prior = beer.JointIsotropicNormalGammaPrior(self.means, self.scales,
                                                         self.shape, self.rate)

    def test_natural2std(self):
        means, scales, shape, rate = \
            self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(means.numpy(), self.means.numpy())
        self.assertArraysAlmostEqual(scales.numpy(), self.scales.numpy())
        self.assertArraysAlmostEqual(shape.numpy(), self.shape.numpy())
        self.assertArraysAlmostEqual(rate.numpy(), self.rate.numpy())

    def test_std2natural(self):
        means, scales, shape, rate = self.prior.to_std_parameters()
        nparams = self.prior.to_natural_parameters(means, scales, shape, rate)
        self.assertArraysAlmostEqual(nparams.numpy(),
                                     self.prior.natural_parameters.numpy())


__all__ = [
    'TestDirichletPrior',
    'TestGammaPrior',
    'TestNormalFullCovariancePrior',
    'TestIsotropicNormalGammaPrior',
    'TestJointIsotropicNormalGammaPrior',
    'TestNormalGammaPrior',
    'TestWishartPrior'
]
