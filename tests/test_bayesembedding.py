'Test the Normal model.'



import unittest
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.insert(0, './')

import beer
from beer.models.mixture import _expand_labels


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
        T1 = model.sufficient_statistics(self.X)[1]
        state = self.encoder(self.X)
        T2 = self.bayesmodel.sufficient_statistics_from_mean_var(state.mean,
            state.var)
        self.assertTrue(np.allclose(T1.data, T2.data))

    def test_forward(self):
        model = beer.BayesianEmbeddingModel(self.encoder, self.bayesmodel)
        T = model.sufficient_statistics(self.X)
        obj_f1 = model(T, self.labels)
        T_s, T = T
        log_pred = model.bayesian_model.log_predictions(T_s).view(
            model.nsamples, T.size(0), -1)
        log_pred = log_pred.mean(dim=0)
        onehot_labels = _expand_labels(labels,
            len(model.bayesian_model.components)).type(T.type())
        log_p_labels = (onehot_labels * log_pred).sum(dim=-1)
        preds = torch.exp(model.bayesian_model.log_predictions(T))
        nparams = model.bayesian_model.components._expected_nparams_as_matrix().data
        nparams = Variable(onehot_labels @ nparams)
        obj_f2 = log_p_labels - model._state.kl_div(nparams)
        self.assertTrue(np.allclose(obj_f1.data, obj_f2.data))


indim = 2
embedding_dim = 10
outdim = 5
data = torch.randn(20, indim)
labels = torch.zeros(20).long()

structure = nn.Sequential(nn.Linear(indim, embedding_dim))
encoder1 = beer.MLPNormalDiag(structure, embedding_dim)
encoder2 = beer.MLPNormalIso(structure, embedding_dim)

prior_w = beer.DirichletPrior(torch.ones(outdim))
post_w = beer.DirichletPrior(torch.ones(outdim))
prior = beer.NormalGammaPrior(torch.zeros(embedding_dim),
    torch.ones(embedding_dim), 1.)
posts = [beer.NormalGammaPrior(torch.randn(embedding_dim),
                               torch.ones(embedding_dim), 1.)
         for _ in range(outdim)]
nset = beer.NormalDiagonalCovarianceSet(prior, posts)
model1 = beer.Mixture(prior_w, post_w, nset)


tests = [
    # Model: Normal (full. cov.).
    (TestBayesianEmbeddingModel, {
            'encoder': encoder1,
            'bayesmodel': model1,
            'X': data,
            'labels': labels
    }),
    (TestBayesianEmbeddingModel, {
            'encoder': encoder2,
            'bayesmodel': model1,
            'X': data,
            'labels': labels
    }),
]


module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))

if __name__ == '__main__':
    unittest.main()

