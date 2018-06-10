
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
from basetest import BaseTest


# pylint: disable=R0902
class TestNormalDiagonalCovariance(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim).type(self.type)
        self.diag_cov = 1 + torch.randn(self.dim).type(self.type) ** 2
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.model = beer.NormalDiagonalCovariance.create(self.mean,
                                                          self.diag_cov,
                                                          self.pseudo_counts)

    def test_create(self):
        mean1, mean2 = self.mean.numpy(), self.model.mean.numpy()
        self.assertArraysAlmostEqual(mean1, mean2)
        cov1, cov2 = np.diag(self.diag_cov.numpy()), self.model.cov.numpy()
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
        stats = self.model.sufficient_statistics(self.data)
        nparams = self.model.parameters[0].expected_value()
        exp_llh1 = stats @ nparams
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = self.model(stats)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_expected_natural_params(self):
        np1, _ = self.model.expected_natural_params(self.means, self.vars)
        np1 = np1.numpy()
        np2 = self.model.parameters[0].expected_value().numpy()
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
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.model = beer.NormalFullCovariance.create(self.mean,
                                                      self.cov,
                                                      self.pseudo_counts)

    def test_create(self):
        mean1, mean2 = self.mean.numpy(), self.model.mean.numpy()
        self.assertArraysAlmostEqual(mean1, mean2)
        cov1, cov2 = self.cov.numpy(), self.model.cov.numpy()
        self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        data = self.data.numpy()
        stats1 = np.c_[(data[:, :, None] * data[:, None, :]).reshape(self.npoints, -1),
                       data, np.ones(self.npoints), np.ones(self.npoints)]
        stats2 = beer.NormalFullCovariance.sufficient_statistics(self.data)
        self.assertArraysAlmostEqual(stats1, stats2.numpy())

    def test_exp_llh(self):
        stats = self.model.sufficient_statistics(self.data)
        nparams = self.model.parameters[0].expected_value()
        exp_llh1 = stats @ nparams
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = self.model(stats)
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
        self.diag_cov = 1 + torch.randn(self.dim).type(self.type) ** 2
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.model = beer.NormalDiagonalCovarianceSet.create(
            self.mean, self.diag_cov, self.ncomp, self.pseudo_counts,
            noise_std=0.
        )

    def test_create(self):
        self.assertEqual(len(self.model), self.ncomp)
        for i in range(self.ncomp):
            mean1, mean2 = self.mean.numpy(), self.model[i].mean.numpy()
            self.assertArraysAlmostEqual(mean1, mean2)
            cov1, cov2 = self.diag_cov.numpy(), torch.diag(self.model[i].cov).numpy()
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
        matrix1 = self.model.expected_natural_params_as_matrix()
        matrix2 = torch.cat([param.expected_value()[None]
                             for param in self.model.parameters])
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        matrix = torch.cat([param.expected_value()[None]
                            for param in self.model.parameters], dim=0)
        T = self.model.sufficient_statistics(self.data)
        exp_llh1 = T @ matrix.t()
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = self.model(T)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())
        T = self.model.sufficient_statistics(self.data)
        acc_stats1 = list(weights.t() @ T)
        acc_stats2 = [value for key, value in self.model.accumulate(T, weights).items()]
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
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.model = beer.NormalFullCovarianceSet.create(
            self.mean, self.cov, self.ncomp, self.pseudo_counts,
            noise_std=0.
        )

    def test_create(self):
        self.assertEqual(len(self.model), self.ncomp)
        for i in range(self.ncomp):
            mean1, mean2 = self.mean.numpy(), self.model[i].mean.numpy()
            self.assertArraysAlmostEqual(mean1, mean2)
            cov1, cov2 = self.cov.numpy(), self.model[i].cov.numpy()
            self.assertArraysAlmostEqual(cov1, cov2)

    def test_sufficient_statistics(self):
        stats1 = beer.NormalFullCovariance.sufficient_statistics(self.data)
        stats2 = beer.NormalFullCovarianceSet.sufficient_statistics(self.data)
        self.assertArraysAlmostEqual(stats1.numpy(), stats2.numpy())

    # pylint: disable=C0103
    def test_expected_natural_params_as_matrix(self):
        matrix1 = self.model.expected_natural_params_as_matrix()
        matrix2 = torch.cat([param.expected_value()[None]
                             for param in self.model.parameters])
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        matrix = torch.cat([param.expected_value()[None]
                            for param in self.model.parameters], dim=0)
        T = self.model.sufficient_statistics(self.data)
        exp_llh1 = T @ matrix.t()
        exp_llh1 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        exp_llh2 = self.model(T)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())
        T = self.model.sufficient_statistics(self.data)
        acc_stats1 = list(weights.t() @ T)
        acc_stats2 = [value for key, value in self.model.accumulate(T, weights).items()]
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
        self.diag_cov = 1 + (torch.randn(self.dim)**2).type(self.type)
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.prior_mean = torch.randn(self.dim).type(self.type)
        self.model = beer.NormalSetSharedDiagonalCovariance.create(
            self.prior_mean, self.diag_cov, self.ncomp, self.pseudo_counts,
            noise_std=0.
        )

    def test_create(self):
        self.assertEqual(len(self.model), self.ncomp)
        for i, comp in enumerate(self.model):
            mean1, mean2 = self.prior_mean.numpy(), comp.mean.numpy()
            self.assertArraysAlmostEqual(mean1, mean2)
            cov1, cov2 = np.diag(self.diag_cov.numpy()), comp.cov.numpy()
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
        matrix1 = self.model.expected_natural_params_as_matrix()
        post = self.model.means_prec_param.posterior
        param1, param2, param3, param4 = \
            post.split_sufficient_statistics(post.expected_sufficient_statistics)
        ones = torch.ones_like(param2)
        matrix2 = torch.cat([
            ones * param1[None, :],
            param2,
            param3,
            ones * param4[None, :]], dim=1)
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        stats1, stats2 = self.model.sufficient_statistics(self.data)
        # pylint: disable=W0212
        params = self.model._expected_nparams()
        exp_llh1 = self.model((stats1, stats2))
        self.assertEqual(exp_llh1.size(0), self.data.size(0))
        self.assertEqual(exp_llh1.size(1), self.ncomp)

        exp_llh2 = (stats1 @ params[0])[:, None] + stats2 @ params[1].t()
        exp_llh2 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())
        feadim = self.data.size(1)
        stats = self.model.sufficient_statistics(self.data)
        acc_stats1 = self.model.accumulate(stats, weights)[self.model.means_prec_param]
        self.assertEqual(len(acc_stats1),
                         len(self.model.means_prec_param.posterior.natural_hparams))
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
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.prior_mean = torch.randn(self.dim).type(self.type)
        self.model = beer.NormalSetSharedFullCovariance.create(
            self.prior_mean, self.cov, self.ncomp, self.pseudo_counts,
            noise_std=0.
        )

    def test_create(self):
        self.assertEqual(len(self.model), self.ncomp)
        for i, comp in enumerate(self.model):
            mean1, mean2 = self.prior_mean.numpy(), comp.mean.numpy()
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
        matrix1 = self.model.expected_natural_params_as_matrix()
        post = self.model.means_prec_param.posterior
        param1, param2, param3, param4 = \
            post.split_sufficient_statistics(post.expected_sufficient_statistics)
        ones1 = torch.ones(self.ncomp, self.dim**2).type(param2.type())
        ones2 = torch.ones(self.ncomp, 1).type(param2.type())
        matrix2 = torch.cat([
            ones1 * param1.view(-1)[None, :],
            param2,
            param3[:, None],
            ones2 * param4], dim=1)
        self.assertArraysAlmostEqual(matrix1.numpy(), matrix2.numpy())

    def test_forward(self):
        stats1, stats2 = self.model.sufficient_statistics(self.data)
        exp_llh1 = self.model((stats1, stats2))
        self.assertEqual(exp_llh1.size(0), self.data.size(0))
        self.assertEqual(exp_llh1.size(1), self.ncomp)

        # pylint: disable=W0212
        params = self.model._expected_nparams()
        exp_llh2 = (stats1 @ params[0])[:, None] + stats2 @ params[1].t() + params[2]
        exp_llh2 -= .5 * self.data.size(1) * math.log(2 * math.pi)
        self.assertArraysAlmostEqual(exp_llh1.numpy(), exp_llh2.numpy())

    def test_accumulate(self):
        weights = torch.ones(len(self.data), self.ncomp).type(self.data.type())
        T = self.model.sufficient_statistics(self.data)
        acc_stats1 = self.model.accumulate(T, weights)[self.model.means_prec_param]
        self.assertEqual(len(acc_stats1),
                         len(self.model.means_prec_param.posterior.natural_hparams))
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

#__all__ = ['TestNormalDiagonalCovariance', 'TestNormalFullCovariance',
#           'TestNormalDiagonalCovarianceSet', 'TestNormalFullCovarianceSet',
#           'TestNormalSetSharedDiagonalCovariance',
#           'TestNormalSetSharedFullCovariance']
