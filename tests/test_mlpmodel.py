'Test the Normal model.'


import unittest
import numpy as np
import math
import torch
import torch.nn as nn

import sys
sys.path.insert(0, './')

import beer
from beer.models.mlpmodel import normal_diag_natural_params
from beer.models.mlpmodel import _structure_output_dim


torch.manual_seed(10)


TOLPLACES = 5
TOL = 10 ** (-TOLPLACES)


class TestMLPModelFunctions:

    def test_normal_diag_nparams(self):
        nparams1 = normal_diag_natural_params(self.mean, self.var)
        nparams2 = torch.cat([
            - 1. / (2 * self.var),
            self.mean / self.var,
            - (self.mean ** 2) / (2 * self.var),
            -.5 * torch.log(self.var)
        ], dim=-1)
        self.assertTrue(np.allclose(nparams1.numpy(), nparams2.numpy(), atol=TOL))

    def test_structure_output_dim(self):
        s_out_dim = _structure_output_dim(self.structure)
        self.assertEqual(s_out_dim, self.s_out_dim)


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
    'vars': torch.randn(20, 2).double() ** 2
}


tests = [
    (TestMLPModelFunctions, {'mean': torch.randn(10, 2).float(),
        'var': (torch.randn(10, 2)**2).float(),
        'structure': nn.Sequential(nn.Linear(2, 10), nn.Tanh()),
        's_out_dim': 10}),
    (TestMLPModelFunctions, {'mean': torch.randn(10, 2).double(),
        'var': (torch.randn(10, 2)**2).double(),
        'structure': nn.Sequential(nn.Linear(2, 10), nn.Linear(10, 1)),
        's_out_dim': 1}),
    (TestMLPModelFunctions, {'mean': torch.randn(10, 2).float(),
        'var': (torch.randn(10, 2)**2).float(),
        'structure': nn.Sequential(nn.Linear(2, 10), nn.Tanh(),
                                   nn.Linear(10, 10), nn.Linear(10, 50)),
        's_out_dim': 50}),
    (TestMLPModelFunctions, {'mean': torch.randn(10, 2).double(),
        'var': (torch.randn(10, 2)**2).double(),
        'structure': nn.Sequential(nn.Linear(2, 10), nn.Tanh(),
                                   nn.Linear(10, 20)),
        's_out_dim': 20}),
]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

