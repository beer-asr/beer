'Test the VAE module.'


# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import numpy as np
from scipy.special import logsumexp, gammaln
import torch
import beer
from basetest import BaseTest


class TestVAE(BaseTest):

    def setUp(self):
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.nsamples = int(1 + torch.randint(20, (1, 1)).item())

    def test_normal_log_likelihood(self):
        data = torch.randn(self.npoints, self.dim)
        mean = torch.randn(self.nsamples, self.npoints, self.dim).type(self.type)
        variance = 1 + torch.randn(self.nsamples,
                                   self.npoints, self.dim).type(self.type) ** 2
        llh1 = beer.vae._normal_log_likelihood(data, mean, variance)

        data, mean, variance = data.numpy(), mean.numpy(), variance.numpy()
        llh2 = -.5 * (((data[None] - mean) ** 2) / variance).sum(axis=-1)
        llh2 -= .5 * np.log(variance).sum(axis=-1)
        llh2 = llh2.mean(axis=0)
        llh2 -= .5 * self.dim * np.log(2 * np.pi)

        self.assertArraysAlmostEqual(llh1.numpy(), llh2)

    def test_bernoulli_log_likelihood(self):
        m = torch.distributions.Bernoulli(torch.tensor([1. / self.dim] * self.dim))
        data = m.sample().type(self.type)
        mean = torch.randn(self.nsamples, self.npoints, self.dim).type(self.type)
        mean = torch.nn.functional.sigmoid(mean)
        llh1 = beer.vae._bernoulli_log_likelihood(data, mean)

        data, mean = data.numpy(), mean.numpy()
        epsilon = 1e-6
        llh2 = data[None] * np.log(epsilon + mean) + \
            (1 - data[None]) * np.log(epsilon + 1 - mean)
        llh2 = llh2.sum(axis=-1).mean(axis=0)

        self.assertArraysAlmostEqual(llh1.numpy(), llh2)

    def test_beta_log_likelihood(self):
        alpha = torch.tensor([1. / self.dim] * self.dim)
        beta = torch.tensor([1. / self.dim] * self.dim)
        m = torch.distributions.Beta(alpha, beta)
        data = m.sample().type(self.type)
        alpha = torch.randn(self.nsamples, self.npoints, self.dim).type(self.type)
        alpha = torch.nn.functional.sigmoid(alpha)
        beta = torch.randn(self.nsamples, self.npoints, self.dim).type(self.type)
        beta = torch.nn.functional.sigmoid(beta)
        llh1 = beer.vae._beta_log_likelihood(data, alpha, beta)

        data, alpha, beta = data.numpy(), alpha.numpy(), beta.numpy()
        epsilon = 1e-6
        llh2 = (alpha - 1) * np.log(epsilon + data[None]) + \
            (beta - 1) * np.log(epsilon + 1 - data[None])
        llh2 += gammaln(alpha + beta) - \
            gammaln(alpha) - gammaln(beta)
        llh2 = llh2.sum(axis=-1).mean(axis=0)

        self.assertArraysAlmostEqual(llh1.numpy(), llh2)


__all__ = ['TestVAE']
