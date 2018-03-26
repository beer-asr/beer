'Test the Normal model.'



import unittest
import numpy as np
import math
import torch
import torch.nn as nn

import sys
sys.path.insert(0, './')

import beer


torch.manual_seed(10)


TOLPLACES = 5
TOL = 10 ** (-TOLPLACES)


class TestBayesianEmbeddingModel:

    def test_create(self):
        model = beer.BayesianEmbeddingModel(self.encoder, self.bayesmodel)
        self.assertTrue(isinstance(model, beer.BayesianModel))
        for param1, param2 in zip(model.parameters, self.bayesmodel.parameters):
            self.assertIs(param1, param2)

    def test_sufficient_ststatistics(self):
        model = beer.BayesianEmbeddingModel(self.encoder, self.bayesmodel)
        T1 = model.sufficient_statistics(self.X)
        state = self.encoder(self.X)
        T2 = self.bayesmodel.sufficient_statistics_from_mean_var(state.mean,
            state.var)
        self.assertTrue(np.allclose(T1.data, T2.data))

    def test_forward(self):
        model = beer.BayesianEmbeddingModel(self.encoder, self.bayesmodel)
        T = model.sufficient_statistics(self.X)
        exp_llh1 = model(T, self.labels)
        state = self.encoder(self.X)
        exp_llh2 = self.bayesmodel(T, self.labels) + state.entropy()
        self.assertTrue(np.allclose(exp_llh1.data, exp_llh2.data))


indim = 2
embedding_dim = 10
outdim = 5
data = torch.randn(20, indim)
labels = torch.zeros(20).long()

structure = nn.Sequential(nn.Linear(indim, embedding_dim))
encoder1 = beer.MLPNormalDiag(structure, outdim)
encoder2 = beer.MLPNormalIso(structure, outdim)

model1 = beer.NormalDiagonalCovariance(
    beer.NormalGammaPrior(torch.zeros(outdim), torch.ones(outdim), 1.),
    beer.NormalGammaPrior(torch.zeros(outdim), torch.ones(outdim), 1.),
)
model2 = beer.NormalFullCovariance(
    beer.NormalWishartPrior(torch.zeros(outdim), torch.eye(outdim), 1.),
    beer.NormalWishartPrior(torch.zeros(outdim), torch.eye(outdim), 1.),
)

prior_w = beer.DirichletPrior(torch.ones(24))
post_w = beer.DirichletPrior(torch.ones(24))
prior = beer.NormalGammaPrior(torch.zeros(outdim), torch.ones(outdim), 1.)
posts = [beer.NormalGammaPrior(torch.randn(2), torch.ones(2), 1.)
         for _ in range(10)]
nset = beer.NormalDiagonalCovarianceSet(prior, posts)
model3 = beer.Mixture(prior_w, post_w, nset)


prior_w = beer.DirichletPrior(torch.ones(24))
post_w = beer.DirichletPrior(torch.ones(24))
prior = beer.NormalWishartPrior(torch.zeros(outdim), torch.eye(outdim), 1.)
earosts = [beer.NormalWishartPrior(torch.randn(2), torch.eye(2), 1.)
         for _ in range(10)]
nset = beer.NormalFullCovarianceSet(prior, posts)
model4 = beer.Mixture(prior_w, post_w, nset)


tests = [
    # Model: Normal (diag. cov.).
    (TestBayesianEmbeddingModel, {
            'encoder': encoder1,
            'bayesmodel': model1,
            'X': data,
            'labels': labels
    }),
    (TestBayesianEmbeddingModel, {
            'encoder': encoder1,
            'bayesmodel': model1,
            'X': data,
            'labels': None
    }),
    (TestBayesianEmbeddingModel, {
            'encoder': encoder2,
            'bayesmodel': model1,
            'X': data,
            'labels': labels
    }),
    (TestBayesianEmbeddingModel, {
            'encoder': encoder2,
            'bayesmodel': model1,
            'X': data,
            'labels': None
    }),

    # Model: Normal (full. cov.).
    (TestBayesianEmbeddingModel, {
            'encoder': encoder1,
            'bayesmodel': model2,
            'X': data,
            'labels': labels
    }),
    (TestBayesianEmbeddingModel, {
            'encoder': encoder1,
            'bayesmodel': model2,
            'X': data,
            'labels': None
    }),
    (TestBayesianEmbeddingModel, {
            'encoder': encoder2,
            'bayesmodel': model2,
            'X': data,
            'labels': labels
    }),
    (TestBayesianEmbeddingModel, {
            'encoder': encoder2,
            'bayesmodel': model2,
            'X': data,
            'labels': None
    }),
]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

