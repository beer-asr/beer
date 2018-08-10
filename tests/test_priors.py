'Test the priors package.'

import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import unittest
import torch
import beer
from basetest import BaseTest


########################################################################
# Dirichlet.
########################################################################

class TestDirichletPrior(BaseTest):

    def setUp(self):
        dim = 10
        self.std_parameters = 2 * torch.ones(dim)
        self.prior = beer.DirichletPrior(self.std_parameters)

    def test_natural2std(self):
        std_params = self.prior.to_std_parameters(self.prior.natural_parameters)
        self.assertArraysAlmostEqual(
            std_params.numpy(),
            self.std_parameters.numpy()
        )

    def test_std2natural(self):
        std_params = self.prior.to_std_parameters(self.prior.natural_parameters)
        nparams = self.prior.to_natural_parameters(std_params)
        self.assertArraysAlmostEqual(
            std_params.numpy(),
            self.std_parameters.numpy()
        )

    def test_exp_sufficient_statistics(self):
        pass

    def test_log_norm(self):
        pass


__all__ = [
    'TestDirichletPrior',
]
