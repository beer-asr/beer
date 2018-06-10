'Test the Normal model.'

import sys
sys.path.insert(0, './')

import torch
import beer
from basetest import BaseTest

# Even though the VB-EM algorithm is theoretically guaranteed to
# increase, it may happen in practice due to floating point precision
# issue that it decreases a little bit at one step. Settings TOLERANCE
# to 0 will make the test fails if one of such update occurs.
TOLERANCE = 1e-6

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

        self.mean = torch.zeros(self.dim).type(self.type)
        self.prec = 2
        self.subspace = torch.randn(self.dim - 1, self.dim).type(self.type)
        self.ppca = beer.PPCA.create(self.mean, self.prec, self.subspace)

        ncomps = int(2 + torch.randint(10, (1, 1)).item())
        obs_dim = self.dim
        noise_s_dim = self.dim - 1
        class_s_dim = ncomps - 1
        mean = torch.zeros(self.dim).type(self.type)
        prec = 1.
        noise_s = torch.randn(noise_s_dim, obs_dim).type(self.type)
        class_s = torch.randn(class_s_dim, obs_dim).type(self.type)
        means = 2 * torch.randn(ncomps, class_s_dim).type(self.type)
        weights = torch.ones(ncomps).type(self.type) / ncomps
        pseudo_counts = 1.

        pldaset = beer.PLDASet.create(mean, prec, noise_s, class_s, means, pseudo_counts)
        self.plda = beer.Mixture.create(weights, pldaset, pseudo_counts)

    def test_optim1(self):
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
                    elbo = round(float(elbo) / (len(self.data) * self.dim), 3)
                    self.assertGreaterEqual(elbo - previous, -TOLERANCE)
                    previous = elbo

    def test_optim2(self):
        elbo_fn = beer.EvidenceLowerBound(len(self.data))
        for i, model in enumerate([self.ppca, self.plda]):
            with self.subTest(i=i):
                optim = beer.BayesianModelCoordinateAscentOptimizer(
                    *model.grouped_parameters, lrate=1.)
                previous = -float('inf')
                for _ in range(100):
                    optim.zero_grad()
                    elbo = elbo_fn(model, self.data)
                    elbo.natural_backward()
                    optim.step()
                    elbo = round(float(elbo) / (len(self.data) * self.dim), 3)
                    self.assertGreaterEqual(elbo - previous, -TOLERANCE)
                    previous = elbo

__all__ = ['TestEvidenceLowerbound']
