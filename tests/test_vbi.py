'Test the Normal model.'

import unittest


import sys
sys.path.insert(0, './')

import torch
import beer
from basetest import BaseTest


class TestEvidenceLowerbound(BaseTest):

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)

        self.models = []
        self.models += [
            beer.NormalDiagonalCovariance.create(
                torch.zeros(self.dim).type(self.type),
                torch.ones(self.dim).type(self.type)
            ),
            beer.NormalFullCovariance.create(
                torch.zeros(self.dim).type(self.type),
                torch.eye(self.dim).type(self.type)
            ),
        ]

        normalset = beer.NormalDiagonalCovarianceSet.create(
            torch.zeros(self.dim).type(self.type),
            torch.ones(self.dim).type(self.type),
            10,
            noise_std=0.1
        )
        weights = torch.ones(10).type(self.type) * .1
        self.models.append(beer.Mixture.create(weights, normalset))

        normalset = beer.NormalFullCovarianceSet.create(
            torch.zeros(self.dim).type(self.type),
            torch.eye(self.dim).type(self.type),
            10,
            noise_std=0.1
        )
        weights = torch.ones(10).type(self.type) * .1
        self.models.append(beer.Mixture.create(weights, normalset))

        normalset = beer.NormalSetSharedDiagonalCovariance.create(
            torch.zeros(self.dim).type(self.type),
            torch.ones(self.dim).type(self.type),
            10,
            noise_std=0.1
        )
        weights = torch.ones(10).type(self.type) * .1
        self.models.append(beer.Mixture.create(weights, normalset))

        normalset = beer.NormalSetSharedFullCovariance.create(
            torch.zeros(self.dim).type(self.type),
            torch.eye(self.dim).type(self.type),
            10,
            noise_std=0.1
        )
        weights = torch.ones(10).type(self.type) * .1
        self.models.append(beer.Mixture.create(weights, normalset))

        normalset = beer.NormalSetSharedFullCovariance.create(
            torch.zeros(self.dim).type(self.type),
            torch.eye(self.dim).type(self.type),
            2,
            noise_std=0.1
        )
        self.models.append(beer.HMM.create([0, 1], [0, 1], 
                                           torch.FloatTensor([[1, 0], 
                                            [.5, .5]]).type(self.type),
                                            normalset))


    def test_elbo(self):
        elbo_fn = beer.EvidenceLowerBound(len(self.data))
        for i, model in enumerate(self.models):
            with self.subTest(i=i):
                optim = beer.BayesianModelOptimizer(model.parameters, lrate=1.)
                previous = -float('inf')
                for _ in range(100):
                    optim.zero_grad()
                    elbo = elbo_fn(model, self.data)
                    elbo.natural_backward()
                    optim.step()

                    elbo = round(float(elbo) / len(self.data), self.tolplaces)
                    self.assertGreaterEqual(elbo, previous)
                    previous = elbo

__all__ = ['TestEvidenceLowerbound']
