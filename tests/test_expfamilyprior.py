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
# JointExpFamilyPrior.
########################################################################

class TestJointExpFamilyPrior(BaseTest):

    def setUp(self):
        dim = int(1 + torch.randint(100, (1, 1)).item())
        self.concentrations = (torch.randn(dim) ** 2).type(self.type)
        self.prior1 = beer.DirichletPrior(self.concentrations)
        self.concentrations = (torch.randn(dim) ** 2).type(self.type)
        self.prior2 = beer.DirichletPrior(self.concentrations)
        self.prior = beer.JointExpFamilyPrior([self.prior1, self.prior2])

    def test_init(self):
        nhp1 = self.prior.natural_hparams.numpy()
        nhp2 = np.r_[self.prior1.natural_hparams.numpy(),
                     self.prior2.natural_hparams.numpy()]
        self.assertArraysAlmostEqual(nhp1, nhp2)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.prior, self.prior)), 0.)

    def test_exp_sufficient_statistics(self):
        stats1 = self.prior.expected_sufficient_statistics.numpy()
        stats2 = np.r_[self.prior1.expected_sufficient_statistics.numpy(),
                       self.prior2.expected_sufficient_statistics.numpy()]
        self.assertArraysAlmostEqual(stats1, stats2)

    def test_log_norm(self):
        lnorm1 = self.prior.log_norm(self.prior.natural_hparams).numpy()
        lnorm2 = self.prior1.log_norm(self.prior1.natural_hparams).numpy()
        lnorm2 += self.prior2.log_norm(self.prior2.natural_hparams).numpy()
        self.assertAlmostEqual(lnorm1, lnorm2, places=self.tolplaces)

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
        self.concentrations = (torch.randn(dim) ** 2).type(self.type)
        self.model = beer.DirichletPrior(self.concentrations)

    def test_init(self):
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     self.concentrations.numpy() - 1)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_exp_sufficient_statistics(self):
        model_s_stats = self.model.expected_sufficient_statistics.numpy()
        natural_params = self.model.natural_hparams.numpy()
        s_stats = dirichlet_grad_log_norm(natural_params)
        self.assertArraysAlmostEqual(model_s_stats, s_stats)

    def test_log_norm(self):
        model_log_norm = self.model.log_norm(self.model.natural_hparams).numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm = dirichlet_log_norm(natural_hparams)
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
        self.scale = (1 + torch.randn(self.dim)**2).type(self.type)
        self.shape = (1 + torch.randn(self.dim)**2).type(self.type)
        self.rate = (1 + torch.randn(self.dim)**2).type(self.type)
        self.model = beer.NormalGammaPrior(self.mean, self.scale,
                                           self.shape, self.rate)

    def test_init(self):
        mean, scale, shape, rate = self.mean.numpy(), self.scale.numpy(), \
            self.shape.numpy(), self.rate.numpy()
        natural_hparams = np.hstack([
            scale * (mean ** 2) + 2 * rate,
            scale * mean,
            scale,
            2 * shape - 1
        ])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_hparams)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_exp_sufficient_statistics(self):
        model_s_stats = self.model.expected_sufficient_statistics.numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        s_stats = normalgamma_grad_log_norm(natural_hparams)
        self.assertArraysAlmostEqual(model_s_stats, s_stats)

    def test_log_norm(self):
        log_norm1 = self.model.log_norm(self.model.natural_hparams).numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = normalgamma_log_norm(natural_hparams)
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


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
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.means = torch.randn((self.ncomp, self.dim)).type(self.type)
        self.scales = (1 + torch.randn((self.ncomp, self.dim))**2).type(
            self.type)
        self.shape = (1 + torch.randn(self.dim)**2).type(self.type)
        self.rate = (1 + torch.randn(self.dim)**2).type(self.type)
        self.model = beer.JointNormalGammaPrior(self.means, self.scales,
                                                self.shape, self.rate)

    def test_init(self):
        means, scales, shape, rate = self.means.numpy(), self.scales.numpy(), \
            self.shape.numpy(), self.rate.numpy()
        natural_hparams = np.hstack([
            ((scales * means**2).sum(axis=0) + 2 * rate).reshape(-1),
            (scales * means).reshape(-1),
            (scales).reshape(-1),
            2 * shape - 1
        ])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_hparams)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_log_norm(self):
        log_norm1 = self.model.log_norm(self.model.natural_hparams).numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = jointnormalgamma_log_norm(natural_hparams, ncomp=self.ncomp)
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


########################################################################
# Normal Wishart prior.
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
        self.scale = (1 + torch.randn(1)**2).type(self.type)
        scale_mat = (1 + torch.randn(self.dim)).type(self.type)
        self.scale_mat = torch.eye(self.dim).type(self.type) + \
            torch.ger(scale_mat, scale_mat)
        self.dof = (self.dim + 100 + torch.randn(1)**2).type(self.type)
        self.model = beer.NormalWishartPrior(self.mean, float(self.scale),
                                             self.scale_mat, float(self.dof))

    def test_init(self):
        mean, scale, scale_mat, dof = self.mean.numpy(), self.scale.numpy(), \
            self.scale_mat.numpy(), self.dof.numpy()
        natural_hparams = np.hstack([
            (scale * np.outer(mean, mean) + np.linalg.inv(scale_mat)).reshape(-1),
            scale * mean,
            self.scale,
            dof - self.dim
        ])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_hparams)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_exp_sufficient_statistics(self):
        s_stats1 = self.model.expected_sufficient_statistics.numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        s_stats2 = normalwishart_grad_log_norm(natural_hparams)
        self.assertArraysAlmostEqual(s_stats1, s_stats2)

    def test_log_norm(self):
        log_norm1 = self.model.log_norm(self.model.natural_hparams).numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = normalwishart_log_norm(natural_hparams)
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


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
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.means = torch.randn((self.ncomp, self.dim)).type(self.type)
        self.scales = (1 + torch.randn((self.ncomp))**2).type(self.type)
        scale_mat = (1 + torch.randn(self.dim)).type(self.type)
        self.scale_mat = torch.eye(self.dim).type(self.type) + \
            torch.ger(scale_mat, scale_mat)
        self.dof = (self.dim + 100 + torch.randn(1)**2).type(self.type)
        self.model = beer.JointNormalWishartPrior(self.means, self.scales,
                                                  self.scale_mat,
                                                  float(self.dof))

    def test_init(self):
        means, scales, scale_mat, dof = self.means.numpy(), self.scales.numpy(), \
            self.scale_mat.numpy(), self.dof.numpy()
        inv_scale = np.linalg.inv(scale_mat)
        quad_mean = ((scales[:, None] * means)[:, None, :] * means[:, :, None]).sum(axis=0)
        natural_params = np.hstack([
            (quad_mean + inv_scale).reshape(-1),
            (scales[:, None] * means).reshape(-1),
            scales,
            dof - self.dim
        ])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_params)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_log_norm(self):
        log_norm1 = self.model.log_norm(self.model.natural_hparams).numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = jointnormalwishart_log_norm(natural_hparams,
                                                ncomp=self.ncomp)
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


########################################################################
# Normal prior (full cov).
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


class TestNormalFullCovariancePrior(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim).type(self.type)
        cov = (1 + torch.randn(self.dim)).type(self.type)
        self.cov = torch.eye(self.dim).type(self.type) + torch.ger(cov, cov)
        self.model = beer.NormalFullCovariancePrior(self.mean, self.cov)

    def test_init(self):
        mean, cov = self.mean.numpy(), self.cov.numpy()
        prec = np.linalg.inv(cov)
        natural_hparams = np.hstack([
            -.5 * prec.reshape(-1),
            prec @ mean,
        ])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_hparams)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_exp_sufficient_statistics(self):
        s_stats1 = self.model.expected_sufficient_statistics.numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        s_stats2 = normal_fc_grad_log_norm(natural_hparams)
        self.assertArraysAlmostEqual(s_stats1, s_stats2)

    def test_log_norm(self):
        log_norm1 = self.model.log_norm(self.model.natural_hparams).numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = normal_fc_log_norm(natural_hparams)[0]
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


########################################################################
# Normal prior (isotropic cov).
########################################################################

def normal_iso_split_np(natural_params):
    return natural_params[0], natural_params[1:]


def normal_iso_log_norm(natural_params):
    np1, np2 = normal_iso_split_np(natural_params)
    inv_np1 = 1 / np1
    logdet = np.log(-2 * np1)
    lognorm = -.5 * len(np2) * logdet - .25 * inv_np1 * (np2[None, :] @ np2)
    return lognorm


def normal_iso_grad_log_norm(natural_params):
    np1, np2 = normal_iso_split_np(natural_params)
    variance = 1 / (-2 * np1)
    mean = variance * np2
    return np.hstack([len(np2) * variance + np.sum(mean ** 2), mean])


class TestNormalIsotropicCovariancePrior(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim).type(self.type)
        self.var = (1 + torch.randn(self.dim)**2).type(self.type)
        self.model = beer.NormalIsotropicCovariancePrior(self.mean,
                                                         self.var)

    def test_init(self):
        mean, var = self.mean.numpy(), self.var.numpy()
        prec = 1 / var
        natural_hparams = np.hstack([
            -.5 * prec,
            prec * mean,
        ])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_hparams)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_exp_sufficient_statistics(self):
        s_stats1 = self.model.expected_sufficient_statistics.numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        s_stats2 = normal_iso_grad_log_norm(natural_hparams)
        self.assertArraysAlmostEqual(s_stats1, s_stats2)

    def test_log_norm(self):
        log_norm1 = float(self.model.log_norm(self.model.natural_hparams))
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = float(normal_iso_log_norm(natural_hparams))
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


########################################################################
# Matrix Normal prior.
########################################################################

def matrixnormal_fc_split_np(natural_params, dim1, dim2):
    np1, np2 = natural_params[:dim1 ** 2], natural_params[dim1 ** 2:]
    return np1.reshape(dim1, dim1), np2.reshape(dim1, dim2)


def matrixnormal_fc_log_norm(natural_params, dim1, dim2):
    np1, np2 = matrixnormal_fc_split_np(natural_params, dim1, dim2)
    inv_np1 = np.linalg.inv(np1)
    sign, logdet = np.linalg.slogdet(-2 * np1)
    lognorm = -.5 * dim2 * sign * logdet - .25 * np.trace(np2.T @ inv_np1 @ np2)
    return lognorm.item()


def matrixnormal_fc_grad_log_norm(natural_params, dim1, dim2):
    np1, np2 = matrixnormal_fc_split_np(natural_params, dim1, dim2)
    cov = np.linalg.inv(-2 * np1)
    mean = cov @ np2
    return np.hstack([(dim2 * cov + mean @ mean.T).reshape(-1),
                      mean.reshape(-1)])


class TestMatrixNormalPrior(BaseTest):

    def setUp(self):
        self.dim1 = int(1 + torch.randint(100, (1, 1)).item())
        self.dim2 = int(1 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim1, self.dim2).type(self.type)
        cov = (1 + torch.randn(self.dim1)).type(self.type)
        self.cov = torch.eye(self.dim1).type(self.type) + torch.ger(cov, cov)
        self.model = beer.MatrixNormalPrior(self.mean, self.cov)

    def test_init(self):
        mean, cov = self.mean.numpy(), self.cov.numpy()
        prec = np.linalg.inv(cov)
        natural_hparams = np.hstack([
            -.5 * prec.reshape(-1),
            (prec @ mean).reshape(-1),
        ])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_hparams)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_exp_sufficient_statistics(self):
        s_stats1 = self.model.expected_sufficient_statistics.numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        s_stats2 = matrixnormal_fc_grad_log_norm(natural_hparams, self.dim1,
                                                 self.dim2)
        self.assertArraysAlmostEqual(s_stats1, s_stats2)

    def test_log_norm(self):
        log_norm1 = self.model.log_norm(self.model.natural_hparams).numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = matrixnormal_fc_log_norm(natural_hparams, self.dim1,
                                             self.dim2)
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


########################################################################
# Gamma prior.
########################################################################

def gamma_log_norm(natural_params):
    return float(gammaln(natural_params[0] + 1) - \
        (natural_params[0] + 1) * np.log(-natural_params[1]))


def gamma_grad_log_norm(natural_params):
    return np.hstack([psi(natural_params[0] + 1) - np.log(-natural_params[1]),
                      (natural_params[0] + 1) / (-natural_params[1])])


class TestGammaPrior(BaseTest):

    def setUp(self):
        self.shape = (1 + torch.randn(1) ** 2).type(self.type)
        self.rate = (1 + torch.randn(1) ** 2).type(self.type)
        self.model = beer.GammaPrior(self.shape, self.rate)

    def test_init(self):
        shape, rate = self.shape.numpy(), self.rate.numpy()
        natural_hparams = np.hstack([shape - 1, -rate])
        self.assertArraysAlmostEqual(self.model.natural_hparams.numpy(),
                                     natural_hparams)

    def test_kl_div(self):
        self.assertAlmostEqual(float(beer.ExpFamilyPrior.kl_div(
            self.model, self.model)), 0.)

    def test_exp_sufficient_statistics(self):
        s_stats1 = self.model.expected_sufficient_statistics.numpy()
        natural_hparams = self.model.natural_hparams.numpy()
        s_stats2 = gamma_grad_log_norm(natural_hparams)
        self.assertArraysAlmostEqual(s_stats1, s_stats2)

    def test_log_norm(self):
        log_norm1 = float(self.model.log_norm(self.model.natural_hparams).numpy())
        natural_hparams = self.model.natural_hparams.numpy()
        log_norm2 = gamma_log_norm(natural_hparams)
        self.assertAlmostEqual(log_norm1, log_norm2, places=self.tolplaces)


__all__ = [
    'TestDirichletPrior', 'TestNormalGammaPrior', 'TestJointNormalGammaPrior',
    'TestNormalWishartPrior', 'TestJointNormalWishartPrior',
    'TestNormalFullCovariancePrior', 'TestNormalIsotropicCovariancePrior',
    'TestMatrixNormalPrior', 'TestGammaPrior', 'TestJointExpFamilyPrior'
]
