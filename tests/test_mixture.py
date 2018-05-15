'Test the Mixture model.'


# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import numpy as np
from scipy.special import logsumexp
import torch
import beer
from beer.models.mixture import _expand_labels
from basetest import BaseTest


def create_normalset_diag(ncomps, dim, type_t):
    posts = [beer.NormalGammaPrior(torch.zeros(dim).type(type_t),
                                   torch.ones(dim).type(type_t),
                                   1.)
             for _ in range(ncomps)]
    normalset = beer.NormalDiagonalCovarianceSet(
        beer.NormalGammaPrior(torch.zeros(dim).type(type_t),
                              torch.ones(dim).type(type_t),
                              1.),
        posts
    )
    return normalset


def create_normalset_full(ncomps, dim, type_t):
    posts = [beer.NormalWishartPrior(torch.zeros(dim).type(type_t),
                                     torch.eye(dim).type(type_t),
                                     1.)
             for _ in range(ncomps)]
    normalset = beer.NormalFullCovarianceSet(
        beer.NormalWishartPrior(torch.zeros(dim).type(type_t),
                                torch.eye(dim).type(type_t),
                                1.),
        posts
    )
    return normalset


# pylint: disable=R0902
class TestGMMDiag(BaseTest):

    def setUp(self):
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.normalset = create_normalset_diag(self.ncomp, self.dim, self.type)
        self.prior_counts = torch.ones(self.ncomp).type(self.type)
        self.dir_prior = beer.DirichletPrior(self.prior_counts)
        self.dir_posterior = beer.DirichletPrior(self.prior_counts)

    def test_create(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        self.assertEqual(len(model.components), self.ncomp)

    def test_sufficient_statistics(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        stats1 = model.sufficient_statistics(self.data).numpy()
        stats2 = model.components.sufficient_statistics(self.data).numpy()
        self.assertArraysAlmostEqual(stats1, stats2)

    # pylint: disable=C0103
    def test_sufficient_statistics_from_mean_var(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        stats1 = model.sufficient_statistics_from_mean_var(
            self.data,
            torch.ones_like(self.data)
        ).numpy()
        stats2 = model.components.sufficient_statistics_from_mean_var(
            self.data,
            torch.ones_like(self.data)
        ).numpy()
        self.assertArraysAlmostEqual(stats1, stats2)

    def test_forward(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        stats = model.sufficient_statistics(self.data)
        pc_exp_llh = (model.components(stats) + \
            self.dir_posterior.expected_sufficient_statistics.view(1, -1))
        pc_exp_llh = pc_exp_llh.numpy()
        exp_llh1 = logsumexp(pc_exp_llh, axis=1)
        exp_llh2 = model(stats).numpy()
        self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_exp_llh_labels(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        labels = torch.zeros(self.data.size(0)).long()
        elabels = _expand_labels(labels, len(model.components))
        mask = torch.log(elabels).numpy()
        elabels = elabels.numpy()
        stats = model.sufficient_statistics(self.data)
        pc_exp_llh = (model.components(stats) + \
            self.dir_posterior.expected_sufficient_statistics.view(1, -1))
        pc_exp_llh = pc_exp_llh.numpy()
        pc_exp_llh += mask
        exp_llh1 = logsumexp(pc_exp_llh, axis=1)
        exp_llh2 = model(stats, labels).numpy()
        self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_expand_labels(self):
        ref = torch.range(0, 2).long()
        labs1 = _expand_labels(ref, 3).long()
        labs2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertArraysAlmostEqual(labs1.numpy(), labs2)


# pylint: disable=R0902
class TestGMMFull(BaseTest):

    def setUp(self):
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).double() ** 2
        self.vars = self.vars.type(self.type)
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.normalset = create_normalset_full(self.ncomp, self.dim, self.type)
        self.prior_counts = torch.ones(self.ncomp).type(self.type)
        self.dir_prior = beer.DirichletPrior(self.prior_counts)
        self.dir_posterior = beer.DirichletPrior(self.prior_counts)

    def test_create(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        self.assertEqual(len(model.components), self.ncomp)

    def test_sufficient_statistics(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        stats1 = model.sufficient_statistics(self.data).numpy()
        stats2 = model.components.sufficient_statistics(self.data).numpy()
        self.assertArraysAlmostEqual(stats1, stats2)

    def test_forward(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        stats = model.sufficient_statistics(self.data)
        pc_exp_llh = (model.components(stats) + \
            self.dir_posterior.expected_sufficient_statistics.view(1, -1))
        pc_exp_llh = pc_exp_llh.numpy()
        exp_llh1 = logsumexp(pc_exp_llh, axis=1)
        exp_llh2 = model(stats).numpy()
        self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_exp_llh_labels(self):
        model = beer.Mixture(self.dir_prior, self.dir_posterior, self.normalset)
        labels = torch.zeros(self.data.size(0)).long()
        elabels = _expand_labels(labels, len(model.components))
        mask = torch.log(elabels).numpy()
        elabels = elabels.numpy()
        stats = model.sufficient_statistics(self.data)
        pc_exp_llh = (model.components(stats) + \
            self.dir_posterior.expected_sufficient_statistics.view(1, -1))
        pc_exp_llh = pc_exp_llh.numpy()
        pc_exp_llh += mask
        exp_llh1 = logsumexp(pc_exp_llh, axis=1)
        exp_llh2 = model(stats, labels).numpy()
        self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_expand_labels(self):
        ref = torch.range(0, 2).long()
        labs1 = _expand_labels(ref, 3).long()
        labs2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertArraysAlmostEqual(labs1.numpy(), labs2)


__all__ = ['TestGMMDiag', 'TestGMMFull']
