'Test the Mixture model.'


import sys
sys.path.insert(0, './')
import unittest
import numpy as np
from scipy.special import logsumexp
import math
import beer
import torch

from beer.models.mixture import _expand_labels

TOLPLACES = 4
TOL = 10 ** (-TOLPLACES)


class TestMixture:

    def test_create(self):
        model = beer.Mixture(
            beer.DirichletPrior(self.prior_counts),
            beer.DirichletPrior(self.prior_counts),
            self.normalset
        )
        self.assertEqual(len(model.components), len(self.prior_counts))
        w1 = model.weights.numpy()
        w2 = np.ones(len(self.prior_counts)) / len(self.prior_counts)
        self.assertTrue(np.allclose(w1, w2))

    def test_sufficient_statistics(self):
        model = beer.Mixture(
            beer.DirichletPrior(self.prior_counts),
            beer.DirichletPrior(self.prior_counts),
            self.normalset
        )
        s1 = model.sufficient_statistics(self.X)
        s2 = model.components.sufficient_statistics(self.X)
        self.assertTrue(np.allclose(s1.numpy(), s2.numpy()))

    def test_sufficient_statistics_from_mean_var(self):
        model = beer.Mixture(
            beer.DirichletPrior(self.prior_counts),
            beer.DirichletPrior(self.prior_counts),
            self.normalset
        )
        try:
            s1 = model.sufficient_statistics_from_mean_var(self.X,
                torch.ones_like(self.X))
            s2 = model.components.sufficient_statistics_from_mean_var(self.X,
                torch.ones_like(self.X))
            self.assertTrue(np.allclose(s1.numpy(), s2.numpy()))
        except NotImplementedError:
            pass

    def test_forward(self):
        prior = beer.DirichletPrior(self.prior_counts)
        model = beer.Mixture(
            prior, prior,
            self.normalset
        )
        T = model.sufficient_statistics(self.X)
        pc_exp_llh = (model.components(T) + \
            prior.expected_sufficient_statistics.view(1, -1)).numpy()
        exp_llh1 = logsumexp(pc_exp_llh, axis=1)
        exp_llh2  = model(T).numpy()
        self.assertTrue(np.allclose(exp_llh1.astype(exp_llh2.dtype), exp_llh2,
             atol=TOL))

    def test_exp_llh_labels(self):
        prior = beer.DirichletPrior(self.prior_counts)
        model = beer.Mixture(
            prior, prior,
            self.normalset
        )
        labels = torch.zeros(self.X.size(0)).long()
        elabels = _expand_labels(labels, len(model.components))
        mask = torch.log(elabels).numpy()
        elabels = elabels.numpy()
        T = model.sufficient_statistics(self.X)
        pc_exp_llh = (model.components(T) + \
            prior.expected_sufficient_statistics.view(1, -1)).numpy()
        pc_exp_llh += mask
        exp_llh1 = logsumexp(pc_exp_llh, axis=1)
        exp_llh2  = model(T, labels).numpy()
        self.assertTrue(np.allclose(exp_llh1.astype(exp_llh2.dtype), exp_llh2,
             atol=TOL))

    def test_expand_labels(self):
        ref = torch.range(0, 2).long()
        labs1 = _expand_labels(ref, 3).long()
        labs2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(labs1.numpy(), labs2, atol=TOL))

    def test_expected_natural_params(self):
        prior = beer.DirichletPrior(self.prior_counts)
        model = beer.Mixture(
            prior, prior,
            self.normalset
        )

    def test_expected_natural_params(self):
        prior = beer.DirichletPrior(self.prior_counts)
        model = beer.Mixture(
            prior, prior,
            self.normalset
        )

        T = model.sufficient_statistics(self.X)
        np1 = model.expected_natural_params(T)
        matrix = model.components.expected_natural_params_as_matrix()
        pc_exp_llh = (model.components(T) + \
            prior.expected_sufficient_statistics.view(1, -1)).numpy()
        resps = np.exp(pc_exp_llh - logsumexp(pc_exp_llh, axis=1)[:, None])
        np2 = resps @ matrix
        self.assertTrue(np.allclose(np1, np2, atol=TOL))



def create_normalset_diag(ncomps, dim, type_t):
    posts = [beer.NormalGammaPrior(torch.zeros(dim).type(type_t),
                                   torch.ones(dim).type(type_t),
                                   1.)
             for _ in range(ncomps)]
    normalset = beer.NormalDiagonalCovarianceSet(
        beer.NormalGammaPrior(torch.zeros(dim).type(type_t),
                              torch.ones(dim).type(type_t),
                              1.),
        posts
    )
    return normalset


def create_normalset_full(ncomps, dim, type_t):
    posts = [beer.NormalWishartPrior(torch.zeros(dim).type(type_t),
                                     torch.eye(dim).type(type_t),
                                     1.)
             for _ in range(ncomps)]
    normalset = beer.NormalDiagonalCovarianceSet(
        beer.NormalWishartPrior(torch.zeros(dim).type(type_t),
                                torch.eye(dim).type(type_t),
                                1.),
        posts
    )
    return normalset


torch.manual_seed(10)
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

gmm_diag1F = {
    **dataF,
    'prior_counts': torch.ones(10).float(),
    'normalset': create_normalset_diag(10, 2, dataF['X'].type())
}

gmm_diag1D = {
    **dataD,
    'prior_counts': torch.ones(10).double(),
    'normalset': create_normalset_diag(10, 2, dataD['X'].type())

}

gmm_diag2F = {
    **dataF,
    'prior_counts': torch.ones(1).float(),
    'normalset': create_normalset_diag(1, 2, dataF['X'].type())
}

gmm_diag2D = {
    **dataD,
    'prior_counts': torch.ones(1).double(),
    'normalset': create_normalset_diag(1, 2, dataD['X'].type())
}



gmm_full1F = {
    **dataF,
    'prior_counts': torch.ones(10).float(),
    'normalset': create_normalset_full(10, 2, dataF['X'].type())
}

gmm_full1D = {
    **dataD,
    'prior_counts': torch.ones(10).double(),
    'normalset': create_normalset_full(10, 2, dataD['X'].type())
}

gmm_full2F = {
    **dataF,
    'prior_counts': torch.ones(1).float(),
    'comp_type': beer.NormalFullCovariance,
    'normalset': create_normalset_full(1, 2, dataF['X'].type())
}

gmm_full2D = {
    **dataD,
    'prior_counts': torch.ones(1).double(),
    'normalset': create_normalset_full(1, 2, dataD['X'].type())
}


tests = [
    (TestMixture, gmm_diag1F),
    (TestMixture, gmm_diag1D),
    (TestMixture, gmm_diag2F),
    (TestMixture, gmm_diag2D),
    (TestMixture, gmm_full1F),
    (TestMixture, gmm_full1D),
    (TestMixture, gmm_full2F),
    (TestMixture, gmm_full2D),
]

module = sys.modules[__name__]
for i, test in enumerate(tests, start=1):
    name = test[0].__name__ + 'Test' + str(i)
    setattr(module, name, type(name, (unittest.TestCase, test[0]),  test[1]))


if __name__ == '__main__':
    unittest.main()

