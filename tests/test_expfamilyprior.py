'Test the expfamilyprior module.'

# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import numpy as np
from scipy.special import gammaln, psi
import torch
import beer
from basetest import BaseTest


########################################################################
# Dirichlet prior.
########################################################################

def dirichlet_log_norm(natural_params):
    return -gammaln(np.sum(natural_params + 1)) \
        + np.sum(gammaln(natural_params + 1))


def dirichlet_grad_log_norm(natural_params):
    return -psi(np.sum(natural_params + 1)) + psi(natural_params + 1)


class TestDirichletPrior(BaseTest):

    def setUp(self):
        dim = int(1 + torch.randint(100, (1, 1)).item())
        self.prior_counts = (torch.randn(dim) ** 2).type(self.type)

    def test_create(self):
        model = beer.DirichletPrior(self.prior_counts)
        self.assertArraysAlmostEqual(model.natural_params.numpy(),
                                     self.prior_counts.numpy() - 1)

    def test_exp_sufficient_statistics(self):
        model = beer.DirichletPrior(self.prior_counts)
        model_s_stats = model.expected_sufficient_statistics.numpy()
        natural_params = model.natural_params.numpy()
        s_stats = dirichlet_grad_log_norm(natural_params)
        self.assertArraysAlmostEqual(model_s_stats, s_stats)

    def test_kl_divergence(self):
        model1 = beer.DirichletPrior(self.prior_counts)
        model2 = beer.DirichletPrior(self.prior_counts)
        div = beer.kl_div(model1, model2)
        self.assertAlmostEqual(div, 0., places=self.tolplaces)

    def test_log_norm(self):
        model = beer.DirichletPrior(self.prior_counts)
        model_log_norm = model.log_norm.numpy()
        natural_params = model.natural_params.numpy()
        log_norm = dirichlet_log_norm(natural_params)
        self.assertAlmostEqual(model_log_norm, log_norm, places=self.tolplaces)


########################################################################
# Normal-Gamma prior.
########################################################################

def normalgamma_log_norm(natural_params):
    np1, np2, np3, np4 = natural_params.reshape(4, -1)
    lognorm = gammaln(.5 * (np4 + 1))
    lognorm += -.5 * np.log(np3)
    lognorm += -.5 * (np4 + 1) * np.log(.5 * (np1 - ((np2**2) / np3)))
    return lognorm.sum()


def normalgamma_grad_log_norm(natural_params):
    np1, np2, np3, np4 = natural_params.reshape(4, -1)
    grad1 = -(np4 + 1) / (2 * (np1 - ((np2 ** 2) / np3)))
    grad2 = (np2 * (np4 + 1)) / (np3 * np1 - (np2 ** 2))
    grad3 = - 1 / (2 * np3) - ((np2 ** 2) * (np4 + 1)) \
        / (2 * np3 * (np3 * np1 - (np2 ** 2)))
    grad4 = .5 * psi(.5 * (np4 + 1)) \
        - .5 *np.log(.5 * (np1 - ((np2 ** 2) / np3)))
    return np.hstack([grad1, grad2, grad3, grad4])


class TestNormalGammaPrior(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim).type(self.type)
        self.precision = (torch.randn(self.dim)**2).type(self.type)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()

    def test_create(self):
        model = beer.NormalGammaPrior(self.mean, self.precision,
                                      self.prior_count)
        n_mean = self.mean.numpy()
        n_precision = self.prior_count * np.ones_like(n_mean)
        g_shapes = self.precision.numpy() * self.prior_count
        g_rates = self.prior_count
        natural_params = np.hstack([
            n_precision * (n_mean ** 2) + 2 * g_rates,
            n_precision * n_mean,
            n_precision,
            2 * g_shapes - 1
        ])
        self.assertArraysAlmostEqual(model.natural_params.numpy(),
                                     natural_params)

    def test_exp_sufficient_statistics(self):
        model = beer.NormalGammaPrior(self.mean, self.precision,
                                      self.prior_count)
        model_s_stats = model.expected_sufficient_statistics.numpy()
        natural_params = model.natural_params.numpy()
        s_stats = normalgamma_grad_log_norm(natural_params)
        self.assertArraysAlmostEqual(model_s_stats, s_stats)

    def test_kl_divergence(self):
        model1 = beer.NormalGammaPrior(self.mean, self.precision,
                                       self.prior_count)
        model2 = beer.NormalGammaPrior(self.mean, self.precision,
                                       self.prior_count)
        div = beer.kl_div(model1, model2)
        self.assertAlmostEqual(div, 0., places=self.tolplaces)

    def test_log_norm(self):
        model = beer.NormalGammaPrior(self.mean, self.precision,
                                      self.prior_count)
        model_log_norm = model.log_norm.numpy()
        natural_params = model.natural_params.numpy()
        log_norm = normalgamma_log_norm(natural_params)
        self.assertAlmostEqual(model_log_norm, log_norm, places=self.tolplaces)


########################################################################
# Joint Normal-Gamma prior.
########################################################################

def jointnormalgamma_split_nparams(natural_params, ncomp):
    dim = len(natural_params) // (2 + 2 * ncomp)
    np1 = natural_params[:dim]
    np2s = natural_params[dim: dim + dim * ncomp]
    np3s = natural_params[dim + dim * ncomp: dim + 2 * dim * ncomp]
    np4 = natural_params[dim + 2 * dim * ncomp:]
    return np1, np2s, np3s, np4, dim


def jointnormalgamma_log_norm(natural_params, ncomp):
    np1, np2s, np3s, np4, dim = jointnormalgamma_split_nparams(natural_params,
                                                               ncomp)
    lognorm = gammaln(.5 * (np4 + 1)).sum()
    lognorm += -.5 * np.log(np3s).sum()
    tmp = ((np2s ** 2) / np3s).reshape((ncomp, dim))
    lognorm += np.sum(-.5 * (np4 + 1) * \
        np.log(.5 * (np1 - tmp.sum(axis=0))))
    return lognorm


class TestJointNormalGammaPrior(BaseTest):

    def setUp(self):
        self.ncomps = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.means = torch.randn((self.ncomps, self.dim)).type(self.type)
        self.precision = (1 + torch.randn(self.dim)**2).type(self.type)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()

    def test_create(self):
        model = beer.JointNormalGammaPrior(self.means, self.precision,
                                           self.prior_count)
        means, prec, prior_count = self.means.numpy(), \
            self.precision.numpy(), self.prior_count
        dim = self.dim
        ncomps = self.ncomps
        natural_params = np.hstack([
            (prior_count * (means**2).sum(axis=0) + 2 * prior_count).reshape(-1),
            (prior_count * means).reshape(-1),
            (np.ones((ncomps, dim)) * prior_count).reshape(-1),
            2 * prec * prior_count - 1
        ])
        self.assertArraysAlmostEqual(model.natural_params.numpy(),
                                     natural_params)

    def test_kl_divergence(self):
        model1 = beer.JointNormalGammaPrior(self.means, self.precision,
                                            self.prior_count)
        model2 = beer.JointNormalGammaPrior(self.means, self.precision,
                                            self.prior_count)
        div = beer.kl_div(model1, model2)
        self.assertAlmostEqual(float(div), 0., places=self.tolplaces)

    def test_log_norm(self):
        model = beer.JointNormalGammaPrior(self.means, self.precision,
                                           self.prior_count)
        model_log_norm = model.log_norm.numpy()
        natural_params = model.natural_params.numpy()
        log_norm = jointnormalgamma_log_norm(natural_params, ncomp=self.ncomps)
        self.assertAlmostEqual(model_log_norm, log_norm,
                               places=self.tolplaces)


########################################################################
# Joint Normal-Gamma prior.
########################################################################

def normalwishart_split_np(natural_params):
    dim = int(.5 * (-1 + np.sqrt(1 + 4 * len(natural_params[:-2]))))
    np1, np2 = natural_params[:int(dim ** 2)].reshape(dim, dim), \
         natural_params[int(dim ** 2):-2]
    np3, np4 = natural_params[-2:]
    return np1, np2, np3, np4, dim


def normalwishart_log_norm(natural_params):
    np1, np2, np3, np4, dim = normalwishart_split_np(natural_params)
    lognorm = .5 * ((np4 + dim) * dim * np.log(2) - dim * np.log(np3))
    sign, logdet = np.linalg.slogdet(np1 - np.outer(np2, np2) / np3)
    lognorm += -.5 * (np4 + dim) * sign * logdet
    lognorm += np.sum(gammaln(.5 * (np4 + dim + 1 - np.arange(1, dim + 1, 1))))
    return lognorm


def normalwishart_grad_log_norm(natural_params):
    np1, np2, np3, np4, dim = normalwishart_split_np(natural_params)
    outer = np.outer(np2, np2) / np3
    matrix = (np1 - outer)
    sign, logdet = np.linalg.slogdet(matrix)
    inv_matrix = np.linalg.inv(matrix)

    grad1 = -.5 * (np4 + dim) * inv_matrix
    grad2 = (np4 + dim) * inv_matrix @ (np2 / np3)
    grad3 = - dim / (2 * np3) - .5 * (np4 + dim) \
        * np.trace(inv_matrix @ (outer / np3))
    grad4 = .5 * np.sum(psi(.5 * (np4 + dim + 1 - np.arange(1, dim + 1, 1))))
    grad4 += -.5 * sign * logdet + .5 * dim * np.log(2)
    return np.hstack([grad1.reshape(-1), grad2, grad3, grad4])


class TestNormalWishartPrior(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim).type(self.type)
        cov = (1 + torch.randn(self.dim)).type(self.type)
        self.cov = torch.eye(self.dim).type(self.type) + torch.ger(cov, cov)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()

    def test_create(self):
        model = beer.NormalWishartPrior(self.mean, self.cov, self.prior_count)
        mean, cov = self.mean.numpy(), self.cov.numpy()
        dof = self.prior_count + self.dim
        mean_cov = dof * cov
        natural_params = np.hstack([
            (self.prior_count * np.outer(mean, mean) + mean_cov).reshape(-1),
            self.prior_count * mean,
            np.asarray([self.prior_count]),
            np.asarray([dof - self.dim])
        ])
        self.assertArraysAlmostEqual(model.natural_params.numpy(),
                                     natural_params)

    def test_exp_sufficient_statistics(self):
        model = beer.NormalWishartPrior(self.mean, self.cov, self.prior_count)
        model_s_stats = model.expected_sufficient_statistics.numpy()
        natural_params = model.natural_params.numpy()
        s_stats = normalwishart_grad_log_norm(natural_params)
        self.assertArraysAlmostEqual(model_s_stats, s_stats)

    def test_kl_divergence(self):
        model1 = beer.NormalWishartPrior(self.mean, self.cov, self.prior_count)
        model2 = beer.NormalWishartPrior(self.mean, self.cov, self.prior_count)
        div = beer.kl_div(model1, model2)
        self.assertAlmostEqual(div, 0., places=self.tolplaces)

    def test_log_norm(self):
        model = beer.NormalWishartPrior(self.mean, self.cov, self.prior_count)
        model_log_norm = model.log_norm.numpy()
        natural_params = model.natural_params.numpy()
        log_norm = normalwishart_log_norm(natural_params)
        self.assertAlmostEqual(model_log_norm, log_norm,
                               places=self.tolplaces)


########################################################################
# Joint Normal-Whishart prior.
########################################################################

def jointnormalwishart_split_np(natural_params, ncomp=1):
    dim = int(.5 * (-ncomp + np.sqrt(ncomp**2 + \
        4 * len(natural_params[:-(ncomp + 1)]))))
    np1, np2s = natural_params[:int(dim ** 2)].reshape(dim, dim), \
         natural_params[int(dim ** 2):-(ncomp+1)].reshape(ncomp, dim)
    np3s = natural_params[-(ncomp+1): -1]
    np4 = natural_params[-1]
    return np1, np2s, np3s, np4, dim


def jointnormalwishart_log_norm(natural_params, ncomp):
    np1, np2s, np3s, np4, dim = jointnormalwishart_split_np(natural_params, ncomp)
    lognorm = .5 * ((np4 + dim) * dim * np.log(2) - dim * np.log(np3s).sum())
    quad_exp = ((np2s[:, None, :] * np2s[:, :, None]) / \
        np3s[:, None, None]).sum(axis=0)
    sign, logdet = np.linalg.slogdet(np1 - quad_exp)
    lognorm += -.5 * (np4 + dim) * sign * logdet
    lognorm += np.sum(gammaln(.5 * (np4 + dim + 1 - np.arange(1, dim + 1, 1))))
    return lognorm


class TestJointNormalWishartPrior(BaseTest):

    def setUp(self):
        self.ncomps = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.means = torch.randn((self.ncomps, self.dim)).type(self.type)
        cov = (1 + torch.randn(self.dim)).type(self.type)
        self.cov = torch.eye(self.dim).type(self.type) + torch.ger(cov, cov)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()

    def test_create(self):
        model = beer.JointNormalWishartPrior(self.means, self.cov,
                                             self.prior_count)
        means, cov = self.means.numpy(), self.cov.numpy()
        dof = self.prior_count + self.dim
        mean_cov = dof * cov
        quad_mean = (means[:, None, :] * means[:, :, None]).sum(axis=0)
        natural_params = np.hstack([
            (self.prior_count * quad_mean + mean_cov).reshape(-1),
            self.prior_count * means.reshape(-1),
            np.ones(self.ncomps) * self.prior_count,
            np.asarray([dof - self.dim])
        ])
        self.assertArraysAlmostEqual(model.natural_params.numpy(),
                                     natural_params)

    def test_kl_divergence(self):
        model1 = beer.JointNormalWishartPrior(self.means, self.cov, self.prior_count)
        model2 = beer.JointNormalWishartPrior(self.means, self.cov, self.prior_count)
        div = beer.kl_div(model1, model2)
        self.assertAlmostEqual(div, 0., places=self.tolplaces)

    def test_log_norm(self):
        model = beer.JointNormalWishartPrior(self.means, self.cov, self.prior_count)
        model_log_norm = model.log_norm.numpy()
        natural_params = model.natural_params.numpy()
        log_norm = jointnormalwishart_log_norm(natural_params,
                                               ncomp=self.ncomps)
        self.assertAlmostEqual(model_log_norm, log_norm,
                               places=self.tolplaces)

    # We don't test the automatic differentiation of the
    # log-normalizer. As long as the log-normlizer is correct, then
    # pytorch should gives us the right gradient.
    #def test_exp_sufficient_statistics(self):
    #   pass


########################################################################
# Normal prior.
########################################################################

def normal_fc_split_np(natural_params):
    dim = int(.5 * (-1 + np.sqrt(1 + 4 * len(natural_params))))
    np1, np2 = natural_params[:int(dim ** 2)].reshape(dim, dim), \
         natural_params[int(dim ** 2):]
    return np1, np2, dim


def normal_fc_log_norm(natural_params):
    np1, np2, _ = normal_fc_split_np(natural_params)
    inv_np1 = np.linalg.inv(np1)
    sign, logdet = np.linalg.slogdet(-2 * np1)
    lognorm = -.5 * sign * logdet - .25 * (np2[None, :] @ inv_np1) @ np2
    return lognorm


def normal_fc_grad_log_norm(natural_params):
    np1, np2, _ = normal_fc_split_np(natural_params)
    cov = np.linalg.inv(-2 * np1)
    mean = cov @ np2
    return np.hstack([(cov + np.outer(mean, mean)).reshape(-1), mean])


class TestNormalPrior(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim).type(self.type)
        cov = (1 + torch.randn(self.dim)).type(self.type)
        self.cov = torch.eye(self.dim).type(self.type) + torch.ger(cov, cov)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()

    def test_create(self):
        model = beer.NormalPrior(self.mean, self.cov)
        mean, cov = self.mean.numpy(), self.cov.numpy()
        natural_params = np.hstack([
            -.5 * cov.reshape(-1),
            cov @ mean,
        ])
        self.assertArraysAlmostEqual(model.natural_params.numpy(),
                                     natural_params)

    def test_exp_sufficient_statistics(self):
        model = beer.NormalPrior(self.mean, self.cov)
        model_s_stats = model.expected_sufficient_statistics.numpy()
        natural_params = model.natural_params.numpy()
        s_stats = normal_fc_grad_log_norm(natural_params)
        self.assertArraysAlmostEqual(model_s_stats, s_stats)

    def test_kl_divergence(self):
        model1 = beer.NormalPrior(self.mean, self.cov)
        model2 = beer.NormalPrior(self.mean, self.cov)
        div = beer.kl_div(model1, model2)
        self.assertAlmostEqual(div, 0., places=self.tolplaces)

    def test_log_norm(self):
        model = beer.NormalPrior(self.mean, self.cov)
        model_log_norm = model.log_norm.numpy()
        natural_params = model.natural_params.numpy()
        log_norm = normal_fc_log_norm(natural_params)[0]
        self.assertAlmostEqual(model_log_norm, log_norm, places=self.tolplaces)


__all__ = [
    'TestDirichletPrior', 'TestNormalGammaPrior', 'TestJointNormalGammaPrior',
    'TestNormalWishartPrior', 'TestJointNormalWishartPrior', 'TestNormalPrior'
]
