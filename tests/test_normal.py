'Test the Normal model.'



import unittest
import numpy as np
import math
import torch

import sys
sys.path.insert(0, './')

import beer
from beer import NormalGammaPrior
from beer import NormalWishartPrior


torch.manual_seed(10)


TOLPLACES = 4
TOL = 10 ** (-TOLPLACES)


class TestNormalDiagonalCovariance:

    def test_create(self):
        model = beer.NormalDiagonalCovariance(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            NormalGammaPrior(self.mean, self.prec, self.prior_count)
        )
        m1, m2 = self.mean.numpy(), model.mean.numpy()
        self.assertTrue(np.allclose(m1, m2, atol=TOL))
        c1, c2 = (1. / self.prec.numpy()), model.cov.numpy()
        if len(c1.shape) == 1:
            c1 = np.diag(c1)
        self.assertTrue(np.allclose(c1, c2, atol=TOL))

    def test_sufficient_statistics(self):
        X =  self.X.numpy()
        s1 = np.c_[self.X**2, self.X, np.ones_like(X), np.ones_like(X)]
        s2 = beer.NormalDiagonalCovariance.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1, s2.numpy(), atol=TOL))

    def test_sufficient_statistics_from_mean_var(self):
        s1 = beer.NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            self.means, self.vars)
        mean, var = self.means.numpy(), self.vars.numpy()
        s2 = np.c_[mean**2 + var, mean, np.ones_like(mean),
                   np.ones_like(mean)]
        self.assertTrue(np.allclose(s1.numpy(), s2, atol=TOL))

    def test_exp_llh(self):
        model = beer.NormalDiagonalCovariance(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            NormalGammaPrior(self.mean, self.prec, self.prior_count)
        )
        T = model.sufficient_statistics(self.X)
        nparams = model.parameters[0].expected_value
        exp_llh1 = T @ nparams
        exp_llh1 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(T)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy(),
                        atol=TOL))

    #def test_expected_natural_params(self):
    #    model = beer.NormalDiagonalCovariance(
    #        NormalGammaPrior(self.mean, self.prec, self.prior_count),
    #        NormalGammaPrior(self.mean, self.prec, self.prior_count)
    #    )



class TestNormalFullCovariance:

    def test_create(self):
        model = beer.NormalFullCovariance(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            NormalWishartPrior(self.mean, self.cov, self.prior_count)
        )
        m1, m2 = self.mean.numpy(), model.mean.numpy()
        self.assertTrue(np.allclose(m1, m2))
        c1, c2 = self.cov.numpy(), model.cov.numpy()
        self.assertTrue(np.allclose(c1, c2, atol=TOL))

    def test_sufficient_statistics(self):
        X = self.X.numpy()
        s1 = np.c_[(X[:, :, None] * X[:, None, :]).reshape(len(X), -1),
            X, np.ones(len(X)), np.ones(len(X))]
        s2 = beer.NormalFullCovariance.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1, s2.numpy(), atol=TOL))

    def test_exp_llh(self):
        model = beer.NormalFullCovariance(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            NormalWishartPrior(self.mean, self.cov, self.prior_count)
        )
        T = model.sufficient_statistics(self.X)
        nparams = model.parameters[0].expected_value
        exp_llh1 = T @ nparams
        exp_llh1 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(T)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy(),
                        atol=TOL))


class TestNormalDiagonalCovarianceSet:

    def test_create(self):
        posts = [NormalGammaPrior(self.mean, self.prec, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalDiagonalCovarianceSet(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            posts
        )
        self.assertEqual(len(model), self.ncomps)
        for i in range(self.ncomps):
            m1, m2 = self.mean.numpy(), model[i].mean.numpy()
            self.assertTrue(np.allclose(m1, m2, atol=TOL))
            c1, c2 = (1. / self.prec.numpy()), torch.diag(model[i].cov).numpy()
            self.assertTrue(np.allclose(c1, c2, atol=TOL))

    def test_sufficient_statistics(self):
        s1 = beer.NormalDiagonalCovariance.sufficient_statistics(self.X)
        s2 = beer.NormalDiagonalCovarianceSet.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1.numpy(), s2.numpy(), atol=TOL))

    def test_sufficient_statistics_from_mean_var(self):
        s1 = beer.NormalDiagonalCovariance.sufficient_statistics_from_mean_var(
            self.means, self.vars)
        s2 = beer.NormalDiagonalCovarianceSet.sufficient_statistics_from_mean_var(
            self.means, self.vars)
        self.assertTrue(np.allclose(s1.numpy(), s2.numpy(), atol=TOL))

    def test_forward(self):
        posts = [NormalGammaPrior(self.mean, self.prec, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalDiagonalCovarianceSet(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            posts
        )
        matrix = torch.cat([param.expected_value[None]
            for param in model.parameters], dim=0)
        T = model.sufficient_statistics(self.X)
        exp_llh1 = T @ matrix.t()
        exp_llh1 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(T)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy()))

    def test_accumulate(self):
        posts = [NormalGammaPrior(self.mean, self.prec, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalDiagonalCovarianceSet(
            NormalGammaPrior(self.mean, self.prec, self.prior_count),
            posts
        )
        weights = torch.ones(len(self.X), self.ncomps).type(self.X.type())
        T = model.sufficient_statistics(self.X)
        acc_stats1 = list(weights.t() @ T)
        acc_stats2 = [value for key, value in model.accumulate(T, weights).items()]
        for s1, s2 in zip(acc_stats1, acc_stats2):
            self.assertTrue(np.allclose(s1.numpy(), s2.numpy()))


class TestNormalFullCovarianceSet:

    def test_create(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        self.assertEqual(len(model.components), self.ncomps)
        for i in range(self.ncomps):
            m1, m2 = self.mean.numpy(), model.components[i].mean.numpy()
            self.assertTrue(np.allclose(m1, m2))
            c1, c2 = self.cov.numpy(), model.components[i].cov.numpy()
            self.assertTrue(np.allclose(c1, c2, atol=TOL))

    def test_sufficient_statistics(self):
        s1 = beer.NormalFullCovariance.sufficient_statistics(self.X)
        s2 = beer.NormalFullCovarianceSet.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1.numpy(), s2.numpy(), atol=TOL))

    def test_forward(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        matrix = torch.cat([param.expected_value[None]
            for param in model.parameters], dim=0)
        T = model.sufficient_statistics(self.X)
        exp_llh1 = T @ matrix.t()
        exp_llh1 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(T)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy()))

    def test_accumulate(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        weights = torch.ones(len(self.X), self.ncomps).type(self.X.type())
        T = model.sufficient_statistics(self.X)
        acc_stats1 = list(weights.t() @ T)
        acc_stats2 = [value for key, value in model.accumulate(T, weights).items()]
        for s1, s2 in zip(acc_stats1, acc_stats2):
            self.assertTrue(np.allclose(s1.numpy(), s2.numpy()))

class TestNormalFullCovarianceSet:

    def test_create(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        self.assertEqual(len(model), self.ncomps)
        for i in range(self.ncomps):
            m1, m2 = self.mean.numpy(), model[i].mean.numpy()
            self.assertTrue(np.allclose(m1, m2))
            c1, c2 = self.cov.numpy(), model[i].cov.numpy()
            self.assertTrue(np.allclose(c1, c2, atol=TOL))

    def test_sufficient_statistics(self):
        s1 = beer.NormalFullCovariance.sufficient_statistics(self.X)
        s2 = beer.NormalFullCovarianceSet.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1.numpy(), s2.numpy(), atol=TOL))

    def test_forward(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        matrix = torch.cat([param.expected_value[None]
            for param in model.parameters], dim=0)
        T = model.sufficient_statistics(self.X)
        exp_llh1 = T @ matrix.t()
        exp_llh1 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        exp_llh2 = model(T)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy()))

    def test_accumulate(self):
        posts = [NormalWishartPrior(self.mean, self.cov, self.prior_count)
                 for _ in range(self.ncomps)]
        model = beer.NormalFullCovarianceSet(
            NormalWishartPrior(self.mean, self.cov, self.prior_count),
            posts
        )
        weights = torch.ones(len(self.X), self.ncomps).type(self.X.type())
        T = model.sufficient_statistics(self.X)
        acc_stats1 = list(weights.t() @ T)
        acc_stats2 = [value for key, value in model.accumulate(T, weights).items()]
        for s1, s2 in zip(acc_stats1, acc_stats2):
            self.assertTrue(np.allclose(s1.numpy(), s2.numpy()))


class TestNormalSetSharedDiagonalCovariance:

    def test_create(self):
        prior = beer.JointNormalGammaPrior(self.prior_means,
            self.prec, self.prior_count)
        posterior = beer.JointNormalGammaPrior(self.posterior_means,
            self.prec, self.prior_count)
        model = beer.NormalSetSharedDiagonalCovariance(prior, posterior,
            self.ncomps)
        self.assertEqual(len(model), self.ncomps)
        for i, comp in enumerate(model):
            m1, m2 = self.posterior_means[i].numpy(), comp.mean.numpy()
            self.assertTrue(np.allclose(m1, m2, atol=TOL))
            c1, c2 = np.diag(1/self.prec.numpy()), comp.cov.numpy()
            if not np.allclose(c1, c2, atol=TOL):
                print(c1, c2)
            self.assertTrue(np.allclose(c1, c2, atol=TOL))

    def test_sufficient_statistics(self):
        s1 = beer.NormalSetSharedDiagonalCovariance.sufficient_statistics(self.X)
        X = self.X.numpy()
        s2 = np.c_[X**2, np.ones_like(X)], \
            np.c_[X, np.ones_like(X)]
        self.assertTrue(np.allclose(s1[0].numpy(), s2[0], atol=TOL))
        self.assertTrue(np.allclose(s1[1].numpy(), s2[1], atol=TOL))

    def test_sufficient_statistics_from_mean_var(self):
        mean = self.means
        var = self.vars
        s1 = beer.NormalSetSharedDiagonalCovariance.sufficient_statistics_from_mean_var(
            mean, var)
        mean, var = mean.numpy(), var.numpy()
        s2 = np.c_[mean**2 + var, np.ones_like(mean)], \
            np.c_[mean, np.ones_like(mean)]
        self.assertTrue(np.allclose(s1[0].numpy(), s2[0], atol=TOL))
        self.assertTrue(np.allclose(s1[1].numpy(), s2[1], atol=TOL))

    def test_forward(self):
        prior = beer.JointNormalGammaPrior(self.prior_means,
            self.prec, self.prior_count)
        posterior = beer.JointNormalGammaPrior(self.posterior_means,
            self.prec, self.prior_count)
        model = beer.NormalSetSharedDiagonalCovariance(prior, posterior,
            self.ncomps)

        matrix = torch.cat([param.expected_value[None]
            for param in model.parameters], dim=0)
        T1, T2 = model.sufficient_statistics(self.X)
        params = model._expected_nparams()
        exp_llh1 = model((T1, T2))
        self.assertEqual(exp_llh1.size(0), self.X.size(0))
        self.assertEqual(exp_llh1.size(1), self.ncomps)

        exp_llh2 = (T1 @ params[0])[:, None] + T2 @ params[1].t()
        exp_llh2 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy(),
            atol=TOL))

    def test_accumulate(self):
        prior = beer.JointNormalGammaPrior(self.prior_means,
            self.prec, self.prior_count)
        posterior = beer.JointNormalGammaPrior(self.posterior_means,
            self.prec, self.prior_count)
        model = beer.NormalSetSharedDiagonalCovariance(prior, posterior,
            self.ncomps)

        weights = torch.ones(len(self.X), self.ncomps).type(self.X.type())
        feadim = self.X.size(1)
        T = model.sufficient_statistics(self.X)
        acc_stats1 = model.accumulate(T, weights)[model.means_prec_param]
        self.assertEqual(len(acc_stats1),
            len(model.means_prec_param.posterior.natural_params))
        acc_stats2 = torch.cat([T[0][:, :feadim].sum(dim=0), \
            (weights.t() @ T[1][:, :feadim]).view(-1),
            (weights.t() @ T[1][:, feadim:]).view(-1),
            len(self.X) * torch.ones(feadim).type(self.X.type())])
        self.assertTrue(np.allclose(acc_stats1.numpy(), acc_stats2.numpy(),
            atol=TOL))



class TestNormalSetSharedFullCovariance:

    def test_create(self):
        prior = beer.JointNormalWishartPrior(self.prior_means,
            self.cov, self.prior_count)
        posterior = beer.JointNormalWishartPrior(self.posterior_means,
            self.cov, self.prior_count)
        model = beer.NormalSetSharedFullCovariance(prior, posterior, self.ncomps)
        self.assertEqual(len(model), self.ncomps)
        for i, comp in enumerate(model):
            m1, m2 = self.posterior_means[i].numpy(), comp.mean.numpy()
            self.assertTrue(np.allclose(m1, m2, atol=TOL))
            c1, c2 = self.cov.numpy(), comp.cov.numpy()
            self.assertTrue(np.allclose(c1, c2, atol=TOL))

    def test_sufficient_statistics(self):
        s1 = beer.NormalSetSharedFullCovariance.sufficient_statistics(self.X)

        X = self.X.numpy()
        s2 = (X[:, :, None] * X[:, None, :]).reshape(len(X), -1), \
            np.c_[X, np.ones(len(X))]
        self.assertTrue(np.allclose(s1[0].numpy(), s2[0], atol=TOL))
        self.assertTrue(np.allclose(s1[1].numpy(), s2[1], atol=TOL))

    def test_sufficient_statistics_from_mean_var(self):
        mean = self.means
        var = self.vars
        s1 = beer.NormalSetSharedFullCovariance.sufficient_statistics_from_mean_var(
            mean, var)
        mean, var = mean.numpy(), var.numpy()
        idxs = np.identity(mean.shape[1]).reshape(-1) == 1
        XX = (mean[:, :, None] * mean[:, None, :]).reshape(mean.shape[0], -1)
        XX[:, idxs] += var
        s2 = XX, np.c_[mean, np.ones(len(mean))]
        self.assertTrue(np.allclose(s1[0].numpy(), s2[0], atol=TOL))
        self.assertTrue(np.allclose(s1[1].numpy(), s2[1], atol=TOL))

    def test_forward(self):
        prior = beer.JointNormalWishartPrior(self.prior_means,
            self.cov, self.prior_count)
        posterior = beer.JointNormalWishartPrior(self.posterior_means,
            self.cov, self.prior_count)
        model = beer.NormalSetSharedFullCovariance(prior, posterior, self.ncomps)

        T1, T2 = model.sufficient_statistics(self.X)
        exp_llh1 = model((T1, T2))
        self.assertEqual(exp_llh1.size(0), self.X.size(0))
        self.assertEqual(exp_llh1.size(1), self.ncomps)

        params = model._expected_nparams()
        exp_llh2 = (T1 @ params[0])[:, None] + T2 @ params[1].t() + params[2]
        exp_llh2 -= .5 * self.X.size(1) * math.log(2 * math.pi)
        self.assertTrue(np.allclose(exp_llh1.numpy(), exp_llh2.numpy(),
            atol=TOL))

    def test_accumulate(self):
        prior = beer.JointNormalWishartPrior(self.prior_means,
            self.cov, self.prior_count)
        posterior = beer.JointNormalWishartPrior(self.posterior_means,
            self.cov, self.prior_count)
        model = beer.NormalSetSharedFullCovariance(prior, posterior, self.ncomps)
        weights = torch.ones(len(self.X), self.ncomps).type(self.X.type())

        T = model.sufficient_statistics(self.X)
        acc_stats1 = model.accumulate(T, weights)[model.means_prec_param]
        self.assertEqual(len(acc_stats1),
            len(model.means_prec_param.posterior.natural_params))
        acc_stats2 = torch.cat([T[0].sum(dim=0), (weights.t() @ self.X).view(-1),
            weights.sum(dim=0),
            len(self.X) * torch.ones(1).type(self.X.type())])
        self.assertTrue(np.allclose(acc_stats1.numpy(), acc_stats2.numpy(),
            atol=TOL))


dataF = {
    'X': torch.randn(20, 2).float(),
    'means': torch.randn(20, 2).float(),
    'vars': torch.randn(20, 2).float() ** 2
}

dataD = {
    'X': torch.randn(20, 2).double(),
    'means': torch.randn(20, 2).double(),
    'vars': torch.randn(20, 2).double() ** 2
}

data10F = {
    'X': torch.randn(20, 10).float(),
    'means': torch.randn(20, 10).float(),
    'vars': torch.randn(20, 10).float() ** 2
}

data10D = {
    'X': torch.randn(20, 10).double(),
    'means': torch.randn(20, 10).double(),
    'vars': torch.randn(20, 10).double() ** 2
}


tests = [
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(),
        'prec': torch.ones(2).float(), 'prior_count': 1., **dataF}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(),
        'prec': torch.ones(2).double(), 'prior_count': 1., **dataD}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(10).float(), 'prec':
        torch.ones(10).float(), 'prior_count': 1., **data10F}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(10).double(), 'prec':
         torch.ones(10).double(), 'prior_count': 1., **data10D}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'prec':
         torch.ones(2).float(), 'prior_count': 1e-3, **dataF}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'prec':
         torch.ones(2).double(), 'prior_count': 1e-8, **dataD}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'prec':
         torch.ones(2).float() * 1e-2, 'prior_count': 1., **dataF}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).double(), 'prec':
         torch.ones(2).double() * 1e-8, 'prior_count': 1., **dataD}),
    (TestNormalDiagonalCovariance, {'mean': torch.ones(2).float(), 'prec':
         torch.ones(2).float() * 1e2, 'prior_count': 1., **dataF}),

    (TestNormalFullCovariance, {'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float(), 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double(), 'prior_count': 1., **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(),
        'cov': torch.FloatTensor([[2, -1.2], [-1.2, 10.]]).float(),
        'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(),
        'cov': torch.DoubleTensor([[2, -1.2], [-1.2, 10.]]).float(),
        'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(10).float(),
        'cov': torch.eye(10).float(), 'prior_count': 1., **data10F}),
    (TestNormalFullCovariance, {'mean': torch.ones(10).double(),
        'cov': torch.eye(10).double(), 'prior_count': 1., **data10D}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float(), 'prior_count': 1e-3, **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double(), 'prior_count': 1e-7, **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float() * 1e-5, 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double() * 1e-8, 'prior_count': 1., **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float() * 1e2, 'prior_count': 1., **dataF}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., **dataD}),
    (TestNormalFullCovariance, {'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., **dataD}),

    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(),
        'prec': torch.ones(2).float(), 'prior_count': 1., **dataF}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(),
        'prec': torch.ones(2).double(), 'prior_count': 1., **dataD}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(10).float(), 'prec':
        torch.ones(10).float(), 'prior_count': 1., **data10F}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(10).double(), 'prec':
         torch.ones(10).double(), 'prior_count': 1., **data10D}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(), 'prec':
         torch.ones(2).float(), 'prior_count': 1e-3, **dataF}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(), 'prec':
         torch.ones(2).double(), 'prior_count': 1e-8, **dataD}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(), 'prec':
         torch.ones(2).float() * 1e-2, 'prior_count': 1., **dataF}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(), 'prec':
         torch.ones(2).double() * 1e-8, 'prior_count': 1., **dataD}),
    (TestNormalDiagonalCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(), 'prec':
         torch.ones(2).float() * 1e2, 'prior_count': 1., **dataF}),

    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float(), 'prior_count': 1., **dataF}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double(), 'prior_count': 1., **dataD}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(),
        'cov': torch.FloatTensor([[2, -1.2], [-1.2, 10.]]).float(),
        'prior_count': 1., **dataF}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(),
        'cov': torch.DoubleTensor([[2, -1.2], [-1.2, 10.]]).float(),
        'prior_count': 1., **dataF}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(10).float(),
        'cov': torch.eye(10).float(), 'prior_count': 1., **data10F}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(10).double(),
        'cov': torch.eye(10).double(), 'prior_count': 1., **data10D}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float(), 'prior_count': 1e-3, **dataF}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double(), 'prior_count': 1e-7, **dataD}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float() * 1e-5, 'prior_count': 1., **dataF}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double() * 1e-8, 'prior_count': 1., **dataD}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).float(),
        'cov': torch.eye(2).float() * 1e2, 'prior_count': 1., **dataF}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., **dataD}),
    (TestNormalFullCovarianceSet, {'ncomps': 13, 'mean': torch.ones(2).double(),
        'cov': torch.eye(2).double() * 1e8, 'prior_count': 1., **dataD}),

    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'prec': (torch.randn(2)**2).float(),
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'prec': (torch.randn(2)**2).double(),
        'prior_count': 1., **dataD}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'prec': (torch.randn(2)**2).float(),
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'prec': (torch.randn(2)**2).double(),
        'prior_count': 1., **dataD}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 10).float(),
        'posterior_means': torch.randn(13, 10).float(),
        'prec': (torch.randn(10)**2).float(), 'prior_count': 1., **data10F}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 10).double(),
        'posterior_means': torch.randn(13, 10).double(),
        'prec': (torch.randn(10)**2).double(),
        'prior_count': 1., **data10D}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'prec': (torch.randn(2)**2).float(),
        'prior_count': 1e-3, **dataF}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'prec': (torch.randn(2)**2).double(),
        'prior_count': 1e-7, **dataD}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'prec': (torch.randn(2)**2).float(),
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'prec': (torch.randn(2)**2).double(),
        'prior_count': 1., **dataD}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'prec': (1 + torch.randn(2)**2).float(),
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedDiagonalCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'prec': (torch.randn(2)**2).double(),
        'prior_count': 1., **dataD}),

    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'cov': torch.eye(2).float(),
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'cov': torch.eye(2).double(),
        'prior_count': 1., **dataD}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'cov': torch.FloatTensor([[2, -1.2], [-1.2, 10.]]),
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'cov': torch.DoubleTensor([[2, -1.2], [-1.2, 10.]]),
        'prior_count': 1., **dataD}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 10).float(),
        'posterior_means': torch.randn(13, 10).float(),
        'cov': torch.eye(10).float(), 'prior_count': 1., **data10F}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 10).double(),
        'posterior_means': torch.randn(13, 10).double(),
        'cov': torch.eye(10).double(),
        'prior_count': 1., **data10D}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'cov': torch.eye(2).float(),
        'prior_count': 1e-3, **dataF}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'cov': torch.eye(2).double(),
        'prior_count': 1e-7, **dataD}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'cov': torch.eye(2).float() * 1e-5,
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'cov': torch.eye(2).double() * 1e-8,
        'prior_count': 1., **dataD}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).float(),
        'posterior_means': torch.randn(13, 2).float(),
        'cov': torch.eye(2).float() * 1e2,
        'prior_count': 1., **dataF}),
    (TestNormalSetSharedFullCovariance, {'ncomps': 13,
        'prior_means': torch.randn(13, 2).double(),
        'posterior_means': torch.randn(13, 2).double(),
        'cov': torch.eye(2).double() * 1e8,
        'prior_count': 1., **dataD}),
]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

