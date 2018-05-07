'Test the Normal model.'


# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import math
import numpy as np
import torch
import beer
from beer import NormalGammaPrior
from beer import NormalWishartPrior
from beer.expfamilyprior import _jointnormalgamma_split_nparams
from beer.expfamilyprior import _jointnormalwishart_split_nparams
from basetest import BaseTest


# pylint: disable=R0902
class TestNormalDiagonalCovariance(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.mean = torch.randn(self.dim).type(self.type)
        self.prec = (torch.randn(self.dim)**2).type(self.type)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()

    def test_create(self):
        model = beer.NormalDiagonalCovariance(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            NormalGammaPrior(self.mean, self.prec, self.prior_count)
        )
        mean1, mean2 = self.mean.numpy(), model.mean.numpy()
        self.assertArraysAlmostEqual(mean1, mean2)
        cov1, cov2 = (1. / self.prec.numpy()), model.cov.numpy()
        if len(cov1.shape) == 1:
            cov1 = np.diag(cov1)
        self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        data = self.data.numpy()
        stats1 = np.c_[self.data ** 2, self.data, np.ones_like(data),
                       np.ones_like(data)]
        stats2 = beer.NormalDiagonalCovariance.sufficient_statistics(self.data)
        self.assertArraysAlmostEqual(stats1, stats2.numpy())

    # pylint: disable=C0103
    def test_sufficient_statistics_from_mean_var(self):
        stats1 = beer.NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            self.means, self.vars)
        mean, var = self.means.numpy(), self.vars.numpy()
        stats2 = np.c_[mean ** 2 + var, mean, np.ones_like(mean),
                       np.ones_like(mean)]
        self.assertArraysAlmostEqual(stats1.numpy(), stats2)

    def test_exp_llh(self):
        model = beer.NormalDiagonalCovariance(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            NormalGammaPrior(self.mean, self.prec, self.prior_count)
        )
        stats = model.sufficient_statistics(self.data)
        nparams = model.parameters[0].expected_value
        exp_llh1 = stats @ nparams
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(stats)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_expected_natural_params(self):
        model = beer.NormalDiagonalCovariance(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            NormalGammaPrior(self.mean, self.prec, self.prior_count)
        )
        np1 = model.expected_natural_params(self.means, self.vars).numpy()
        np2 = model.parameters[0].expected_value.numpy()
        np2 = np.ones((self.means.size(0), len(np2))) * np2
        self.assertArraysAlmostEqual(np1, np2)


# pylint: disable=R0902
class TestNormalFullCovariance(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.mean = torch.randn(self.dim).type(self.type)
        cov = (1 + torch.randn(self.dim)).type(self.type)
        self.cov = torch.eye(self.dim).type(self.type) + torch.ger(cov, cov)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()

    def test_create(self):
        model = beer.NormalFullCovariance(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            NormalWishartPrior(self.mean, self.cov, self.prior_count)
        )
        mean1, mean2 = self.mean.numpy(), model.mean.numpy()
        self.assertArraysAlmostEqual(mean1, mean2)
        cov1, cov2 = self.cov.numpy(), model.cov.numpy()
        self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        data = self.data.numpy()
        stats1 = np.c_[(data[:, :, None] * data[:, None, :]).reshape(self.npoints, -1),
                       data, np.ones(self.npoints), np.ones(self.npoints)]
        stats2 = beer.NormalFullCovariance.sufficient_statistics(self.data)
        self.assertArraysAlmostEqual(stats1, stats2.numpy())

    def test_exp_llh(self):
        model = beer.NormalFullCovariance(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            NormalWishartPrior(self.mean, self.cov, self.prior_count)
        )
        stats = model.sufficient_statistics(self.data)
        nparams = model.parameters[0].expected_value
        exp_llh1 = stats @ nparams
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(stats)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())


# pylint: disable=R0902
class TestNormalDiagonalCovarianceSet(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.mean = torch.randn(self.dim).type(self.type)
        self.prec = (torch.randn(self.dim)**2).type(self.type)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())

    def test_create(self):
        posts = [NormalGammaPrior(self.mean, self.prec, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalDiagonalCovarianceSet(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            posts
        )
        self.assertEqual(len(model), self.ncomp)
        for i in range(self.ncomp):
            mean1, mean2 = self.mean.numpy(), model[i].mean.numpy()
            self.assertArraysAlmostEqual(mean1, mean2)
            cov1, cov2 = (1. / self.prec.numpy()), torch.diag(model[i].cov).numpy()
            self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        stats1 = beer.NormalDiagonalCovariance.sufficient_statistics(self.data)
        stats2 = beer.NormalDiagonalCovarianceSet.sufficient_statistics(self.data)
        self.assertArraysAlmostEqual(stats1.numpy(), stats2.numpy())

    # pylint: disable=C0103
    def test_sufficient_statistics_from_mean_var(self):
        stats1 = beer.NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            self.means, self.vars)
        stats2 = beer.NormalDiagonalCovarianceSet.sufficient_statistics_from_mean_var(
            self.means, self.vars)
        self.assertArraysAlmostEqual(stats1.numpy(), stats2.numpy())

    def test_expected_natural_params_as_matrix(self):
        posts = [NormalGammaPrior(self.mean, self.prec, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalDiagonalCovarianceSet(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            posts
        )
        matrix1 = model.expected_natural_params_as_matrix()
        matrix2 = torch.cat([param.expected_value[None]
                             for param in model.parameters])
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        posts = [NormalGammaPrior(self.mean, self.prec, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalDiagonalCovarianceSet(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            posts
        )
        matrix = torch.cat([param.expected_value[None]
                            for param in model.parameters], dim=0)
        T = model.sufficient_statistics(self.data)
        exp_llh1 = T @ matrix.t()
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(T)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        posts = [NormalGammaPrior(self.mean, self.prec, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalDiagonalCovarianceSet(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            posts
        )
        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())
        T = model.sufficient_statistics(self.data)
        acc_stats1 = list(weights.t() @ T)
        acc_stats2 = [value for key, value in model.accumulate(T, weights).items()]
        for s1, s2 in zip(acc_stats1, acc_stats2):
            self.assertArraysAlmostEqual(s1.numpy(), s2.numpy())


# pylint: disable=R0902
class TestNormalFullCovarianceSet(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.mean = torch.randn(self.dim).type(self.type)
        cov = (1 + torch.randn(self.dim)).type(self.type)
        self.cov = torch.eye(self.dim).type(self.type) + torch.ger(cov, cov)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())

    def test_create(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        self.assertEqual(len(model), self.ncomp)
        for i in range(self.ncomp):
            mean1, mean2 = self.mean.numpy(), model[i].mean.numpy()
            self.assertArraysAlmostEqual(mean1, mean2)
            cov1, cov2 = self.cov.numpy(), model[i].cov.numpy()
            self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        stats1 = beer.NormalFullCovariance.sufficient_statistics(self.data)
        stats2 = beer.NormalFullCovarianceSet.sufficient_statistics(self.data)
        self.assertArraysAlmostEqual(stats1.numpy(), stats2.numpy())

    # pylint: disable=C0103
    def test_expected_natural_params_as_matrix(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        matrix1 = model.expected_natural_params_as_matrix()
        matrix2 = torch.cat([param.expected_value[None]
                             for param in model.parameters])
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        matrix = torch.cat([param.expected_value[None]
                            for param in model.parameters], dim=0)
        T = model.sufficient_statistics(self.data)
        exp_llh1 = T @ matrix.t()
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(T)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomp)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())
        T = model.sufficient_statistics(self.data)
        acc_stats1 = list(weights.t() @ T)
        acc_stats2 = [value for key, value in model.accumulate(T, weights).items()]
        for s1, s2 in zip(acc_stats1, acc_stats2):
            self.assertArraysAlmostEqual(s1.numpy(), s2.numpy())


# pylint: disable=R0902
class TestNormalSetSharedDiagonalCovariance(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.mean = torch.randn(self.dim).type(self.type)
        self.prec = (torch.randn(self.dim)**2).type(self.type)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.prior_means = torch.randn(self.ncomp, self.dim).type(self.type)
        self.posterior_means = torch.randn(self.ncomp, self.dim).type(self.type)

    def test_create(self):
        prior = beer.JointNormalGammaPrior(self.prior_means,
                                           self.prec, self.prior_count)
        posterior = beer.JointNormalGammaPrior(self.posterior_means,
                                               self.prec, self.prior_count)
        model = beer.NormalSetSharedDiagonalCovariance(prior, posterior,
                                                       self.ncomp)
        self.assertEqual(len(model), self.ncomp)
        for i, comp in enumerate(model):
            mean1, mean2 = self.posterior_means[i].numpy(), comp.mean.numpy()
            self.assertArraysAlmostEqual(mean1, mean2)
            cov1, cov2 = np.diag(1/self.prec.numpy()), comp.cov.numpy()
            self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        stats1 = beer.NormalSetSharedDiagonalCovariance.sufficient_statistics(self.data)
        data = self.data.numpy()
        stats2 = np.c_[self.data ** 2, np.ones_like(data)], \
            np.c_[data, np.ones_like(data)]
        self.assertArraysAlmostEqual(stats1[0].numpy(), stats2[0])
        self.assertArraysAlmostEqual(stats1[1].numpy(), stats2[1])

    # pylint: disable=C0103
    def test_sufficient_statistics_from_mean_var(self):
        mean = self.means
        var = self.vars
        stats1 = beer.NormalSetSharedDiagonalCovariance.sufficient_statistics_from_mean_var(
            mean, var)
        mean, var = mean.numpy(), var.numpy()
        stats2 = np.c_[mean**2 + var, np.ones_like(mean)], \
            np.c_[mean, np.ones_like(mean)]
        self.assertArraysAlmostEqual(stats1[0].numpy(), stats2[0])
        self.assertArraysAlmostEqual(stats1[1].numpy(), stats2[1])

    # pylint: disable=C0103
    def test_expected_natural_params_as_matrix(self):
        prior = beer.JointNormalGammaPrior(self.prior_means,
                                           self.prec, self.prior_count)
        posterior = beer.JointNormalGammaPrior(self.posterior_means,
                                               self.prec, self.prior_count)
        model = beer.NormalSetSharedDiagonalCovariance(prior, posterior,
                                                       self.ncomp)
        matrix1 = model.expected_natural_params_as_matrix()
        nparams = model.parameters[0].expected_value
        param1, param2, param3, param4, _ = _jointnormalgamma_split_nparams(
            nparams, self.ncomp)
        ones = torch.ones_like(param2)
        matrix2 = torch.cat([
            ones * param1[None, :],
            param2,
            param3,
            ones * param4[None, :]], dim=1)
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        prior = beer.JointNormalGammaPrior(self.prior_means,
                                           self.prec, self.prior_count)
        posterior = beer.JointNormalGammaPrior(self.posterior_means,
                                               self.prec, self.prior_count)
        model = beer.NormalSetSharedDiagonalCovariance(prior, posterior,
                                                       self.ncomp)

        stats1, stats2 = model.sufficient_statistics(self.data)
        # pylint: disable=W0212
        params = model._expected_nparams()
        exp_llh1 = model((stats1, stats2))
        self.assertEqual(exp_llh1.size(0), self.data.size(0))
        self.assertEqual(exp_llh1.size(1), self.ncomp)

        exp_llh2 = (stats1 @ params[0])[:, None] + stats2 @ params[1].t()
        exp_llh2 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        prior = beer.JointNormalGammaPrior(self.prior_means,
                                           self.prec, self.prior_count)
        posterior = beer.JointNormalGammaPrior(self.posterior_means,
                                               self.prec, self.prior_count)
        model = beer.NormalSetSharedDiagonalCovariance(prior, posterior,
                                                       self.ncomp)

        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())
        feadim = self.data.size(1)
        stats = model.sufficient_statistics(self.data)
        acc_stats1 = model.accumulate(stats, weights)[model.means_prec_param]
        self.assertEqual(len(acc_stats1),
                         len(model.means_prec_param.posterior.natural_params))
        acc_stats2 = torch.cat([
            stats[0][:, :feadim].sum(dim=0),
            (weights.t() @ stats[1][:, :feadim]).view(-1),
            (weights.t() @ stats[1][:, feadim:]).view(-1),
            len(self.data) * torch.ones(feadim).type(self.data.type())
        ])
        self.assertArraysAlmostEqual(acc_stats1.numpy(), acc_stats2.numpy())


# pylint: disable=R0902
class TestNormalSetSharedFullCovariance(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.mean = torch.randn(self.dim).type(self.type)
        cov = (1 + torch.randn(self.dim)).type(self.type)
        self.cov = torch.eye(self.dim).type(self.type) + torch.ger(cov, cov)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.prior_means = torch.randn(self.ncomp, self.dim).type(self.type)
        self.posterior_means = torch.randn(self.ncomp, self.dim).type(self.type)

    def test_create(self):
        prior = beer.JointNormalWishartPrior(self.prior_means,
                                             self.cov, self.prior_count)
        posterior = beer.JointNormalWishartPrior(self.posterior_means,
                                                 self.cov, self.prior_count)
        model = beer.NormalSetSharedFullCovariance(prior, posterior, self.ncomp)
        self.assertEqual(len(model), self.ncomp)
        for i, comp in enumerate(model):
            mean1, mean2 = self.posterior_means[i].numpy(), comp.mean.numpy()
            self.assertArraysAlmostEqual(mean1, mean2)
            cov1, cov2 = self.cov.numpy(), comp.cov.numpy()
            self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        data = self.data.numpy()
        stats1 = beer.NormalSetSharedFullCovariance.sufficient_statistics(self.data)
        stats2 = (data[:, :, None] * data[:, None, :]).reshape(len(data), -1), \
            np.c_[data, np.ones(len(data))]
        self.assertArraysAlmostEqual(stats1[0].numpy(), stats2[0])
        self.assertArraysAlmostEqual(stats1[1].numpy(), stats2[1])

    # pylint: disable=C0103
    def test_expected_natural_params_as_matrix(self):
        prior = beer.JointNormalWishartPrior(self.prior_means,
                                             self.cov, self.prior_count)
        posterior = beer.JointNormalWishartPrior(self.posterior_means,
                                                 self.cov, self.prior_count)
        model = beer.NormalSetSharedFullCovariance(prior, posterior, self.ncomp)

        matrix1 = model.expected_natural_params_as_matrix()
        nparams = model.parameters[0].expected_value
        param1, param2, param3, param4, D = _jointnormalwishart_split_nparams(
            nparams, self.ncomp)
        ones1 = torch.ones(self.ncomp, D**2).type(param2.type())
        ones2 = torch.ones(self.ncomp, 1).type(param2.type())
        matrix2 = torch.cat([
            ones1 * param1.view(-1)[None, :],
            param2,
            param3[:, None],
            ones2 * param4], dim=1)
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        prior = beer.JointNormalWishartPrior(self.prior_means,
                                             self.cov, self.prior_count)
        posterior = beer.JointNormalWishartPrior(self.posterior_means,
                                                 self.cov, self.prior_count)
        model = beer.NormalSetSharedFullCovariance(prior, posterior, self.ncomp)

        stats1, stats2 = model.sufficient_statistics(self.data)
        exp_llh1 = model((stats1, stats2))
        self.assertEqual(exp_llh1.size(0), self.data.size(0))
        self.assertEqual(exp_llh1.size(1), self.ncomp)

        # pylint: disable=W0212
        params = model._expected_nparams()
        exp_llh2 = (stats1 @ params[0])[:, None] + stats2 @ params[1].t() + params[2]
        exp_llh2 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        prior = beer.JointNormalWishartPrior(self.prior_means,
                                             self.cov, self.prior_count)
        posterior = beer.JointNormalWishartPrior(self.posterior_means,
                                                 self.cov, self.prior_count)
        model = beer.NormalSetSharedFullCovariance(prior, posterior, self.ncomp)
        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())

        T = model.sufficient_statistics(self.data)
        acc_stats1 = model.accumulate(T, weights)[model.means_prec_param]
        self.assertEqual(len(acc_stats1),
                         len(model.means_prec_param.posterior.natural_params))
        acc_stats2 = torch.cat([
            T[0].sum(dim=0), (weights.t() @ self.data).view(-1),
            weights.sum(dim=0),
            len(self.data) * torch.ones(1).type(self.data.type())
        ])
        self.assertArraysAlmostEqual(acc_stats1.numpy(), acc_stats2.numpy())


__all__ = ['TestNormalDiagonalCovariance', 'TestNormalFullCovariance',
           'TestNormalDiagonalCovarianceSet', 'TestNormalFullCovarianceSet',
           'TestNormalSetSharedDiagonalCovariance',
           'TestNormalSetSharedFullCovariance']
