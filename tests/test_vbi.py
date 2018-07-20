'Test the Normal model.'

import sys
sys.path.insert(0, './')
import glob
import os
import unittest
import yaml
import torch
import beer
from basetest import BaseTest

# Even though the VB-EM algorithm is theoretically guaranteed to
# increase, it may happen in practice due to floating point precision
# issue that it decreases a little bit at one step. Settings TOLERANCE
# to 0 will make the test fails if one of such update occurs.
TOLERANCE = 1e-2

# Number of iteration to run while testing the VBI algorithm.
N_EPOCHS = 2
N_ITER = 30


class TestEvidenceLowerbound(BaseTest):

    def setUp(self):
        self.device = 'cpu'
        self.dim = int(10 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.mean = torch.randn(self.dim).type(self.type)
        self.variance = (1 + torch.randn(self.dim) ** 2).type(self.type)

        self.conf_files = []
        for path in glob.glob('./tests/models/*yml'):
            if not os.path.basename(path).startswith('normalset') and \
               not os.path.basename(path).startswith('bernoulli'):
                self.conf_files.append(path)
        assert len(self.conf_files) > 0

        self.models = []
        for conf_file in self.conf_files:
            with open(conf_file, 'r') as fid:
                data = fid.read()
                data = data.replace('<feadim>', str(self.dim))
                conf = yaml.load(data)
            model = beer.create_model(conf, self.mean, self.variance)
            self.models.append(model)

        self.acc_stats1 = {
            'shared_key': torch.randn(self.dim).type(self.type),
            'key1': torch.randn(self.dim).type(self.type),
        }
        self.acc_stats2 = {
            'shared_key': torch.randn(self.dim).type(self.type),
            'key2': torch.randn(self.dim).type(self.type),
        }

    def test_add_acc_stats(self):
        new_stats = beer.vbi.add_acc_stats(self.acc_stats1, self.acc_stats2)
        self.assertArraysAlmostEqual(new_stats['shared_key'].numpy(),
            self.acc_stats1['shared_key'].numpy() + self.acc_stats2['shared_key'].numpy())
        self.assertArraysAlmostEqual(new_stats['key1'].numpy(),
                                     self.acc_stats1['key1'].numpy())
        self.assertArraysAlmostEqual(new_stats['key2'].numpy(),
                                     self.acc_stats2['key2'].numpy())

        new_stats = beer.vbi.add_acc_stats({}, {})
        self.assertEqual(len(new_stats), 0)

    def test_sum(self):
        for i, model in enumerate(self.models):
            with self.subTest(model=self.conf_files[i]):
                optim = beer.BayesianModelCoordinateAscentOptimizer(
                        model.mean_field_groups, lrate=1.)
                previous = -float('inf')
                for _ in range(N_EPOCHS):
                    self.seed(1)
                    optim.zero_grad()
                    elbo = beer.evidence_lower_bound(datasize=len(self.data))
                    for _ in range(N_ITER):
                        elbo += beer.evidence_lower_bound(model, self.data)
                    elbo.natural_backward()
                    optim.step()
                    elbo_val = round(float(elbo) / (len(self.data) * self.dim), 3)
                    self.assertGreaterEqual(elbo_val - previous, -TOLERANCE)
                    previous = elbo_val

    def test_optim(self):
        for i, model in enumerate(self.models):
            with self.subTest(model=self.conf_files[i]):
                optim = beer.BayesianModelCoordinateAscentOptimizer(
                        model.mean_field_groups, lrate=1.)
                previous = -float('inf')
                for _ in range(N_ITER):
                    self.seed(1)
                    optim.zero_grad()
                    elbo = beer.evidence_lower_bound(model, self.data)
                    elbo.natural_backward()
                    optim.step()
                    elbo = round(float(elbo) / (len(self.data) * self.dim), 3)
                    self.assertGreaterEqual(elbo - previous, -TOLERANCE)
                    previous = elbo

    def test_type_switch_float(self):
        for i, orig_model in enumerate(self.models):
            model = orig_model.float()
            with self.subTest(model=self.conf_files[i]):
                optim = beer.BayesianModelCoordinateAscentOptimizer(
                        model.mean_field_groups, lrate=1.)
                previous = -float('inf')
                for _ in range(N_ITER):
                    self.seed(1)
                    optim.zero_grad()
                    elbo = beer.evidence_lower_bound(model, self.data.float())
                    elbo.natural_backward()
                    optim.step()
                    elbo = round(float(elbo) / (len(self.data) * self.dim), 3)
                    self.assertGreaterEqual(elbo - previous, -TOLERANCE)
                    previous = elbo

    def test_type_switch_double(self):
        for i, orig_model in enumerate(self.models):
            model = orig_model.double()
            with self.subTest(model=self.conf_files[i]):
                optim = beer.BayesianModelCoordinateAscentOptimizer(
                        model.mean_field_groups, lrate=1.)
                previous = -float('inf')
                for _ in range(N_ITER):
                    self.seed(1)
                    optim.zero_grad()
                    elbo = beer.evidence_lower_bound(model, self.data.double())
                    elbo.natural_backward()
                    optim.step()
                    elbo = round(float(elbo) / (len(self.data) * self.dim), 3)
                    self.assertGreaterEqual(elbo - previous, -TOLERANCE)
                    previous = elbo

    def test_change_device(self):
        for i, orig_model in enumerate(self.models):
            model = orig_model.to(self.device)
            with self.subTest(model=self.conf_files[i]):
                optim = beer.BayesianModelCoordinateAscentOptimizer(
                        model.mean_field_groups, lrate=1.)
                previous = -float('inf')
                for _ in range(N_ITER):
                    self.seed(1)
                    optim.zero_grad()
                    elbo = \
                        beer.evidence_lower_bound(model, self.data.to(self.device))
                    elbo.natural_backward()
                    optim.step()
                    elbo = round(float(elbo) / (len(self.data) * self.dim), 3)
                    self.assertGreaterEqual(elbo - previous, -TOLERANCE)
                    previous = elbo


__all__ = ['TestEvidenceLowerbound']
