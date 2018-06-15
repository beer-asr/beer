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
from basetest import BaseTest


class TestMixture(BaseTest):

    def setUp(self):
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.weights = (1 + torch.randn(self.ncomp) ** 2).type(self.type)
        self.weights /= self.weights.sum()

        modelsets = [
            beer.NormalDiagonalCovarianceSet.create(
                torch.zeros(self.dim).type(self.type),
                torch.ones(self.dim).type(self.type),
                self.ncomp,
                noise_std=0.1
            ),
            beer.NormalFullCovarianceSet.create(
                torch.zeros(self.dim).type(self.type),
                torch.eye(self.dim).type(self.type),
                self.ncomp,
                noise_std=0.1
            ),
            beer.NormalSetSharedDiagonalCovariance.create(
                torch.zeros(self.dim).type(self.type),
                torch.ones(self.dim).type(self.type),
                self.ncomp,
                noise_std=0.1
            ),
            beer.NormalSetSharedFullCovariance.create(
                torch.zeros(self.dim).type(self.type),
                torch.eye(self.dim).type(self.type),
                self.ncomp,
                noise_std=0.1
            )
        ]

        self.mixtures = []
        for modelset in modelsets:
            self.mixtures.append(beer.Mixture.create(self.weights, modelset,
                                                     self.pseudo_counts))

    def test_create(self):
        for i, mixture in enumerate(self.mixtures):
            with self.subTest(i=i):
                self.assertEqual(len(mixture.modelset), self.ncomp)

    def test_forward(self):
        for i, model in enumerate(self.mixtures):
            with self.subTest(i=i):
                stats = model.sufficient_statistics(self.data)
                pc_exp_llh = (model.modelset(stats) + \
                    model.weights_param.expected_value().view(1, -1))
                pc_exp_llh = pc_exp_llh.numpy()
                exp_llh1 = logsumexp(pc_exp_llh, axis=1)
                exp_llh2 = model(stats).numpy()
                self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_exp_llh_labels(self):
        for i, model in enumerate(self.mixtures):
            with self.subTest(i=i):
                labels = torch.zeros(self.data.size(0)).long()
                elabels = beer.onehot(labels, len(model.modelset),
                                      dtype=self.data.dtype,
                                      device=self.data.device)
                mask = torch.log(elabels).numpy()
                elabels = elabels.numpy()
                stats = model.sufficient_statistics(self.data)
                pc_exp_llh = model.modelset(stats) + \
                    model.weights_param.expected_value().view(1, -1)
                pc_exp_llh = pc_exp_llh.numpy()
                pc_exp_llh += mask
                exp_llh1 = logsumexp(pc_exp_llh, axis=1)
                exp_llh2 = model(stats, labels).numpy()
                self.assertArraysAlmostEqual(exp_llh1, exp_llh2)


__all__ = ['TestMixture']
