'Test the Mixture model.'


# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')
import os
import glob
import yaml
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
        self.mean = torch.randn(self.dim).type(self.type)
        self.variance = 1 + torch.randn(self.dim).type(self.type) ** 2
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.pseudo_counts = 1e-2 + 100 * torch.rand(1).item()
        self.ncomp = int(1 + torch.randint(100, (1, 1)).item())
        self.weights = (1 + torch.randn(self.ncomp) ** 2).type(self.type)
        self.weights /= self.weights.sum()

        self.mixtures = []
        for path in glob.glob('./tests/models/*yaml'):
            if os.path.basename(path).startswith('mixture'):
                with open(path) as fid:
                    conf = yaml.load(fid)
                    model = beer.create_model(conf, self.mean, self.variance)
                self.mixtures.append(model)

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
                exp_llh2 = model(stats, labels=labels).numpy()
                self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_posteriors(self):
        for model in enumerate(self.mixtures):
            with self.subTest(model=model):
                posts = model.posteriors(self.data)
                self.assertAlmostEqual(posts.sum(), len(posts),
                                       places=self.tolplaces)


__all__ = ['TestMixture']
