'Test the arnet module.'

import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')
import numpy as np
import torch

import beer
from basetest import BaseTest


class TestUtils(BaseTest):

    def test_create_mask(self):
        ordering = [3, 1, 2]
        max_connections = [2, 1, 2, 2]
        device = torch.device('cpu')
        mask1 = beer.nnet.arnet.create_mask(ordering, max_connections).numpy()
        mask2 = np.array([
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1]
        ])
        self.assertArraysAlmostEqual(mask1, mask2)

    def test_create_final_mask(self):
        ordering = [3, 1, 2]
        max_connections = [1, 2, 2, 1]
        device = torch.device('cpu')
        mask1 = beer.nnet.arnet.create_final_mask(ordering, max_connections).numpy()
        mask2 = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 1]
        ])
        self.assertArraysAlmostEqual(mask1, mask2)


class TestMaskedLinearTransform(BaseTest):

    def setUp(self):
        self.dim = int(10 + torch.randint(100, (1, 1)).item())
        self.dim_out = int(10 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.ltrans = torch.nn.Linear(self.dim, self.dim_out)
        self.mask = torch.ones(self.dim_out, self.dim)
        self.masked_ltrans = beer.nnet.arnet.MaskedLinear(self.mask, self.ltrans)

    def test_ltrans(self):
        # Don't test the output values, just make sure that everything
        # run.
        out = self.masked_ltrans(self.data)
        self.assertEqual(out.shape[0], self.npoints)
        self.assertEqual(out.shape[1], self.dim_out)

    def test_type_casting(self):
        # Don't test the output values, just make sure that everything
        # run.
        self.masked_ltrans.float()
        self.masked_ltrans.double()
        device = torch.device('cpu')
        self.masked_ltrans.to(device)


class TestARNetwork(BaseTest):

    def setUp(self):
        self.conf_nocontext = {
            'data_dim': int(10 + torch.randint(100, (1, 1)).item()),
            'depth': int(1 + torch.randint(100, (1, 1)).item()),
            'width': int(1 + torch.randint(100, (1, 1)).item()),
            'activation': 'Tanh',
            'context_dim': 0,
        }
        self.conf_context = {
            'data_dim': int(10 + torch.randint(100, (1, 1)).item()),
            'depth': int(1 + torch.randint(100, (1, 1)).item()),
            'width': int(1 + torch.randint(100, (1, 1)).item()),
            'activation': 'Tanh',
            'context_dim': int(1 + torch.randint(100, (1, 1)).item()),
        }

    def test_create(self):
        arnet = beer.nnet.arnet.create_arnetwork(self.conf_nocontext)
        self.assertEqual(len(arnet), 2 * self.conf_nocontext['depth'] + 1)

        arnet = beer.nnet.arnet.create_arnetwork(self.conf_context)
        self.assertEqual(len(arnet), 2 * self.conf_context['depth'] + 1)


__all__ = [
    'TestUtils',
    'TestMaskedLinearTransform',
    'TestARNetwork'
]
