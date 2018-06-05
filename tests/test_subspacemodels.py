'Tests for the subspace models.'

# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import math
import unittest
import numpy as np
import torch
import beer
from basetest import BaseTest

########################################################################
# PPCA.
########################################################################

class TestPPCA(BaseTest):

    def setUp(self):
        self.dim = int(10 + torch.randint(100, (1, 1)).item())
        self.dim_subspace = int(1 + torch.randint(self.dim - 1, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.latent = torch.randn(self.npoints, self.dim_subspace).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2

        # Create the PPCA model
        self.mean = torch.randn(self.dim).type(self.type)
        self.prec = (1 + torch.randn(1) ** 2).type(self.type)
        rand_mat = torch.randn(self.dim_subspace, self.dim)
        q_mat, _  = torch.qr(rand_mat.t())
        self.subspace = q_mat.t().type(self.type)
        self.pseudo_counts = 1e-1 + 100 * torch.rand(1).item()
        self.model = beer.PPCA.create(self.mean, self.prec, self.subspace,
                                      self.pseudo_counts)

    def test_create(self):
        self.assertAlmostEqual(float(self.model.precision), float(self.prec), places=self.tolplaces)
        self.assertArraysAlmostEqual(self.model.mean.numpy(), self.mean)
        self.assertArraysAlmostEqual(self.model.subspace.numpy(), self.subspace)

    def test_kl_div_latent_posteriors(self):
        stats = self.model.sufficient_statistics(self.data)
        l_means, l_cov = self.model.latent_posterior(stats)
        kld1 = beer.PPCA.kl_div_latent_posterior(l_means, l_cov)
        l_means, l_cov = l_means.numpy(), l_cov.numpy()
        s_dim = self.dim_subspace
        sign, logdet = np.linalg.slogdet(l_cov)
        kld2 = .5 * (-sign * logdet + np.trace(l_cov) - s_dim)
        kld2 += .5 * np.sum(l_means**2, axis=1)
        self.assertArraysAlmostEqual(kld1, kld2)

    def test_sufficient_statistics(self):
        data = self.data.numpy()
        stats1 = np.c_[np.sum(data ** 2, axis=1), self.data]
        stats2 = beer.PPCA.sufficient_statistics(self.data)
        self.assertArraysAlmostEqual(stats1, stats2.numpy())

    def test_sufficient_statistics_from_mean_var(self):
        stats1 = beer.PPCA.sufficient_statistics_from_mean_var(self.means,
                                                               self.vars)
        means, variances = self.means.numpy(), self.vars.numpy()
        stats2 = np.c_[np.sum(means ** 2 + variances, axis=1), means]
        self.assertArraysAlmostEqual(stats1.numpy(), stats2)

    def test_latent_posterior(self):
        stats = self.model.sufficient_statistics(self.data)
        data = stats[:, 1:].numpy()
        s_quad, s_mean  =  self.model.subspace_param.expected_value(concatenated=False)
        s_quad, s_mean = s_quad.numpy(), s_mean.numpy()
        prec = self.model.precision.numpy()
        cov1 = np.linalg.inv(np.eye(self.dim_subspace) + prec * s_quad)
        means1 = (prec * cov1 @ s_mean @ (data - self.model.mean.numpy()).T).T
        means2, cov2 = self.model.latent_posterior(stats)
        self.assertArraysAlmostEqual(cov1, cov2.numpy())
        self.assertArraysAlmostEqual(means1, means2.numpy())

    def test_forward(self):
        stats = self.model.sufficient_statistics(self.data)
        exp_llh1 = self.model(stats).numpy()

        l_means, l_cov = self.model.latent_posterior(stats)
        kld = beer.PPCA.kl_div_latent_posterior(l_means, l_cov).numpy()
        l_means, l_cov = l_means.numpy(), l_cov.numpy()
        l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
        log_prec, prec = self.model.precision_param.expected_value(concatenated=False)
        log_prec, prec = log_prec.numpy(), prec.numpy()
        s_quad, s_mean = self.model.subspace_param.expected_value(concatenated=False)
        s_mean, s_quad = s_mean.numpy(), s_quad.numpy()
        m_quad, m_mean = self.model.mean_param.expected_value(concatenated=False)
        m_mean, m_quad = m_mean.numpy(), m_quad.numpy()
        stats = stats.numpy()

        data_mean = stats[:, 1:] - m_mean.reshape(1, -1)

        exp_llh2 = np.zeros(len(stats))
        exp_llh2 += -.5 * self.dim * np.log(2 * np.pi)
        exp_llh2 += .5 * self.dim * log_prec
        exp_llh2 += -.5 * prec * stats[:, 0]
        exp_llh2 += prec * stats[:, 1:] @ m_mean
        exp_llh2 += prec * np.sum((l_means @ s_mean) * data_mean, axis=1)
        exp_llh2 += -.5 * prec * l_quad.reshape(len(stats), -1) @ s_quad.reshape(-1)
        exp_llh2 += -.5 * prec * m_quad

        self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_forward_latent_variables(self):
        stats = self.model.sufficient_statistics(self.data)
        exp_llh1 = self.model(stats, self.latent).numpy()

        l_means = self.latent.numpy()
        l_quad = l_means[:, :, None] * l_means[:, None, :]
        l_quad = l_quad.reshape(len(self.data), -1)
        log_prec, prec = self.model.precision_param.expected_value(concatenated=False)
        log_prec, prec = log_prec.numpy(), prec.numpy()
        s_quad, s_mean = self.model.subspace_param.expected_value(concatenated=False)
        s_mean, s_quad = s_mean.numpy(), s_quad.numpy()
        m_quad, m_mean = self.model.mean_param.expected_value(concatenated=False)
        m_mean, m_quad = m_mean.numpy(), m_quad.numpy()
        stats = stats.numpy()

        data_mean = stats[:, 1:] - m_mean.reshape(1, -1)

        exp_llh2 = np.zeros(len(stats))
        exp_llh2 += -.5 * self.dim * np.log(2 * np.pi)
        exp_llh2 += .5 * self.dim * log_prec
        exp_llh2 += -.5 * prec * stats[:, 0]
        exp_llh2 += prec * stats[:, 1:] @ m_mean
        exp_llh2 += prec * np.sum((l_means @ s_mean) * data_mean, axis=1)
        exp_llh2 += -.5 * prec * l_quad.reshape(len(stats), -1) @ s_quad.reshape(-1)
        exp_llh2 += -.5 * prec * m_quad

        self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_expected_natural_params(self):
        nparams1 = self.model.expected_natural_params(self.means, self.vars).numpy()

        stats = self.model.sufficient_statistics_from_mean_var(self.means, self.vars)
        l_means, l_cov = self.model.latent_posterior(stats)
        l_means, l_cov = l_means.numpy(), l_cov.numpy()
        l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
        l_quad = l_quad.reshape(len(self.data), -1)
        log_prec, prec = self.model.precision_param.expected_value(concatenated=False)
        log_prec, prec = log_prec.numpy(), prec.numpy()
        s_quad, s_mean = self.model.subspace_param.expected_value(concatenated=False)
        s_mean, s_quad = s_mean.numpy(), s_quad.numpy()
        m_quad, m_mean = self.model.mean_param.expected_value(concatenated=False)
        m_mean, m_quad = m_mean.numpy(), m_quad.numpy()

        np1 = -.5 * prec * np.ones((len(stats), self.dim))
        np2 = prec * (l_means @ s_mean + m_mean)
        np3 = np.zeros((len(stats), self.dim))
        np3 += -.5 * prec * (l_quad.reshape(len(stats), -1) @ s_quad.reshape(-1)).reshape(-1, 1)
        np3 += - (prec * l_means @ s_mean @ m_mean).reshape(-1, 1)
        np3 += -.5 * prec * m_quad
        np3 /= self.dim
        np4 = .5 * log_prec * np.ones((len(stats), self.dim))
        nparams2 = np.hstack([np1, np2, np3, np4])

        self.assertEqual(nparams1.shape[0], len(self.means))
        self.assertEqual(nparams1.shape[1], 4 * self.means.shape[1])
        self.assertArraysAlmostEqual(nparams1, nparams2)


class TestPLDA(BaseTest):

    def setUp(self):
        self.dim = int(10 + torch.randint(100, (1, 1)).item())
        self.nclasses = int(1 + torch.randint(20, (1, 1)).item())
        self.dim_subspace1 = int(1 + torch.randint(self.dim - 1, (1, 1)).item())
        self.dim_subspace2 = int(1 + torch.randint(self.dim - 1, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2

        # Create the PLDA model
        self.mean = torch.randn(self.dim).type(self.type)
        self.prec = (1 + torch.randn(1) ** 2).type(self.type)
        rand_mat = torch.randn(self.dim_subspace1, self.dim)
        q_mat, _  = torch.qr(rand_mat.t())
        self.subspace1 = q_mat.t().type(self.type)
        rand_mat = torch.randn(self.dim_subspace2, self.dim)
        q_mat, _  = torch.qr(rand_mat.t())
        self.subspace2 = q_mat.t().type(self.type)
        self.class_means = torch.randn(self.nclasses, self.dim_subspace2).type(self.type)
        self.pseudo_counts = 1e-1 + 100 * torch.rand(1).item()
        self.model = beer.PLDA.create(self.mean, self.prec, self.subspace1,
                                      self.subspace2, self.class_means,
                                      self.pseudo_counts)

    def test_create(self):
        self.assertAlmostEqual(float(self.model.precision), float(self.prec), places=self.tolplaces)
        self.assertArraysAlmostEqual(self.model.mean.numpy(), self.mean)
        self.assertArraysAlmostEqual(self.model.noise_subspace.numpy(),
                                     self.subspace1.numpy())
        self.assertArraysAlmostEqual(self.model.class_subspace.numpy(),
                                     self.subspace2.numpy())
        self.assertArraysAlmostEqual(self.model.class_means.numpy(),
                                     self.class_means.numpy())

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def test_len(self):
        self.assertEqual(len(self.model), len(self.class_means))

    def test_getitem(self):
        for i in range(len(self.model)):
            with self.subTest(i=i):
                prec = self.model.precision.numpy()
                _, s_mean = self.model.noise_subspace_param.expected_value(concatenated=False)
                s_mean = s_mean.numpy()
                s_quad = s_mean.T @ s_mean
                cov = np.eye(self.dim) / prec + s_quad
                normal = self.model[i]
                _, s_mean = self.model.class_subspace_param.expected_value(concatenated=False)
                mean = self.model.mean + s_mean.t() @ self.class_means[i]
                self.assertArraysAlmostEqual(normal.mean.numpy(), mean)
                self.assertArraysAlmostEqual(normal.cov.numpy(), cov)

    def test_expected_natural_params_as_matrix(self):
        _ = self.model.sufficient_statistics(self.data)
        matrix1 = self.model.expected_natural_params_as_matrix().numpy()

        l_means, l_cov = self.model.latent_posterior(self.data)
        l_means, l_cov = l_means.numpy(), l_cov.numpy()
        l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
        log_prec, prec = self.model.precision_param.expected_value(concatenated=False)
        log_prec, prec = log_prec.numpy(), prec.numpy()
        noise_s_quad, noise_s_mean = self.model.noise_subspace_param.expected_value(concatenated=False)
        noise_s_mean, noise_s_quad = noise_s_mean.numpy(), noise_s_quad.numpy()
        class_s_quad, class_s_mean = self.model.class_subspace_param.expected_value(concatenated=False)
        class_s_mean, class_s_quad = class_s_mean.numpy(), class_s_quad.numpy()
        m_quad, m_mean = self.model.mean_param.expected_value(concatenated=False)
        m_mean, m_quad = m_mean.numpy(), m_quad.numpy()
        noise_mean = l_means @ noise_s_mean
        class_mean_mean, class_mean_quad = [], []
        for mean_param in self.model.class_mean_params:
            mean_quad, mean = mean_param.expected_value(concatenated=False)
            class_mean_mean.append(mean.numpy())
            class_mean_quad.append(mean_quad.numpy())

        dim = self.dim
        npoints = len(self.data)
        nparams_matrix = []
        for i in range(self.nclasses):
            class_mean = class_mean_mean[i] @ class_s_mean
            lnorm_quad = np.zeros(npoints)
            lnorm_quad += np.sum(l_quad.reshape(npoints, -1) * \
                noise_s_quad.reshape(-1), axis=1)
            lnorm_quad += class_mean_quad[i].reshape(-1) @ \
                class_s_quad.reshape(-1) + m_quad
            lnorm_quad += 2 * noise_mean @ m_mean

            nparams_matrix.append(np.r_[
                -.5 * (prec / dim) * np.ones(dim),
                class_mean,
                -.5 * prec * lnorm_quad,
                .5 * dim * log_prec,
            ])
        matrix2 = np.r_[nparams_matrix]

        self.assertArraysAlmostEqual(matrix1, matrix2)

    ####################################################################

    @unittest.skip('not implemented')
    def test_kl_div_latent_posteriors(self):
        stats = self.model.sufficient_statistics(self.data)
        l_means, l_cov = self.model.latent_posterior(data)
        kld1 = beer.PPCA.kl_div_latent_posterior(l_means, l_cov)
        l_means, l_cov = l_means.numpy(), l_cov.numpy()
        s_dim = self.dim_subspace
        sign, logdet = np.linalg.slogdet(l_cov)
        kld2 = .5 * (-sign * logdet + np.trace(l_cov) - s_dim)
        kld2 += .5 * np.sum(l_means**2, axis=1)
        self.assertArraysAlmostEqual(kld1, kld2)

    def test_sufficient_statistics(self):
        data = self.data.numpy()
        l_means, l_cov = self.model.latent_posterior(self.data)
        l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
        l_means, l_quad = l_means.numpy(), l_quad.numpy()
        noise_mean = l_means @ self.model.noise_subspace.numpy()
        s_quad, s_mean  = \
            self.model.noise_subspace_param.expected_value(concatenated=False)
        s_quad, s_mean = s_quad.numpy(), s_mean.numpy()
        ls_quad = l_quad.reshape(len(self.data), -1) @ \
            s_quad.reshape(-1)

        broadcasting_array = np.ones_like(data) / data.shape[1]
        tmp = np.sum(data ** 2 - 2 * data * noise_mean, axis=1) + ls_quad

        stats1 = np.c_[
            broadcasting_array * tmp.reshape(len(self.data), 1),
            data - noise_mean,
            np.ones((len(self.data), 2 * self.data.shape[1]))
        ]
        stats2 = self.model.sufficient_statistics(self.data).numpy()
        self.assertArraysAlmostEqual(stats1, stats2)

    def test_sufficient_statistics_from_mean_var(self):
        stats1 = beer.PLDA.sufficient_statistics_from_mean_var(self.means,
                                                               self.vars)
        means, variances = self.means.numpy(), self.vars.numpy()
        stats2 = np.c_[np.sum(means ** 2 + variances, axis=1), means]
        self.assertArraysAlmostEqual(stats1.numpy(), stats2)

    @unittest.skip('not implemented')
    def test_latent_posterior(self):
        stats = self.model.sufficient_statistics(self.data)
        data = stats[:, 1:].numpy()
        s_quad, s_mean  =  self.model.subspace_param.expected_value(concatenated=False)
        s_quad, s_mean = s_quad.numpy(), s_mean.numpy()
        prec = self.model.precision.numpy()
        cov1 = np.linalg.inv(np.eye(self.dim_subspace) + prec * s_quad)
        means1 = (prec * cov1 @ s_mean @ (data - self.model.mean.numpy()).T).T
        means2, cov2 = self.model.latent_posterior(stats)
        self.assertArraysAlmostEqual(cov1, cov2.numpy())
        self.assertArraysAlmostEqual(means1, means2.numpy())

    @unittest.skip('not implemented')
    def test_forward(self):
        stats = self.model.sufficient_statistics(self.data)
        exp_llh1 = self.model(stats).numpy()

        matrix = self.model.expected_natural_params_as_matrix().numpy()
        exp_llh2 = stats @ matrix -.5 * self.dim * np.log(2 * np.pi)

        self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    @unittest.skip('not implemented')
    def test_expected_natural_params(self):
        nparams1 = self.model.expected_natural_params(self.means, self.vars).numpy()

        stats = self.model.sufficient_statistics_from_mean_var(self.means, self.vars)
        l_means, l_cov = self.model.latent_posterior(stats)
        l_means, l_cov = l_means.numpy(), l_cov.numpy()
        l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
        l_quad = l_quad.reshape(len(self.data), -1)
        log_prec, prec = self.model.precision_param.expected_value(concatenated=False)
        log_prec, prec = log_prec.numpy(), prec.numpy()
        s_quad, s_mean = self.model.subspace_param.expected_value(concatenated=False)
        s_mean, s_quad = s_mean.numpy(), s_quad.numpy()
        m_quad, m_mean = self.model.mean_param.expected_value(concatenated=False)
        m_mean, m_quad = m_mean.numpy(), m_quad.numpy()

        np1 = -.5 * prec * np.ones((len(stats), self.dim))
        np2 = prec * (l_means @ s_mean + m_mean)
        np3 = np.zeros((len(stats), self.dim))
        np3 += -.5 * prec * (l_quad.reshape(len(stats), -1) @ s_quad.reshape(-1)).reshape(-1, 1)
        np3 += - (prec * l_means @ s_mean @ m_mean).reshape(-1, 1)
        np3 += -.5 * prec * m_quad
        np3 /= self.dim
        np4 = .5 * log_prec * np.ones((len(stats), self.dim))
        nparams2 = np.hstack([np1, np2, np3, np4])

        self.assertEqual(nparams1.shape[0], len(self.means))
        self.assertEqual(nparams1.shape[1], 4 * self.means.shape[1])
        self.assertArraysAlmostEqual(nparams1, nparams2)

__all__ = ['TestPPCA', 'TestPLDA']
