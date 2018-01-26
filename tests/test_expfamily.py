'Test the expfamily model.'

# (Missing class/method docstring) pylint: disable=C0111
# (Unordered import modules) pylint: disable=C0413
# (Module 'torch' has no 'ones' member) pylint: disable=E1101

import sys
sys.path.append('./')
import unittest
import beer
import numpy as np
from scipy.special import gammaln, psi
import torch
from torch.autograd import Variable


########################################################################
## Computations using numpy for testing.
########################################################################

def dirichlet_log_norm(natural_params):
    return -gammaln(np.sum(natural_params + 1)) \
        + np.sum(gammaln(natural_params + 1))


def dirichlet_grad_log_norm(natural_params):
    return -psi(np.sum(natural_params + 1)) + psi(natural_params + 1)


def normalgamma_log_norm(natural_params):
    np1, np2, np3, np4 = natural_params.reshape(4, -1)
    lognorm = gammaln(.5 * (np4 + 1))
    lognorm += -.5 * np.log(np3)
    lognorm += -.5 * (np4 + 1) * np.log(.5 * (np1 - ((np2**2) / np3)))
    return lognorm.sum()


def normalgamma_grad_log_norm(natural_params):
    np1, np2, np3, np4 = natural_params.reshape(4, -1)
    grad1 = -(np4 + 1) / (2 * (np1 - ((np2 ** 2) / np3)))
    grad2 = (np2 * (np4 + 1)) / (np3 * np1 - (np2 ** 2))
    grad3 = - 1 / (2 * np3) - ((np2 ** 2) * (np4 + 1)) \
        / (2 * np3 * (np3 * np1 - (np2 ** 2)))
    grad4 = .5 * psi(.5 * (np4 + 1)) \
        - .5 *np.log(.5 * (np1 - ((np2 ** 2) / np3)))
    return np.hstack([grad1, grad2, grad3, grad4])


def normalwishart_split_np(natural_params):
    D = int(.5 * (-1 + np.sqrt(1 + 4 * len(natural_params[:-2]))))
    np1, np2 = natural_params[:int(D**2)].reshape(D, D), \
         natural_params[int(D**2):-2]
    np3, np4 = natural_params[-2:]
    return np1, np2, np3, np4, D


def normalwishart_log_norm(natural_params):
    np1, np2, np3, np4, D = normalwishart_split_np(natural_params)
    lognorm = .5 * ((np4 + D) * D * np.log(2) - D * np.log(np3))
    sign, logdet = np.linalg.slogdet(np1 - np.outer(np2, np2) / np3)
    lognorm += -.5 * (np4 + D) * sign * logdet
    lognorm += np.sum(gammaln(.5 * (np4 + D + 1 - np.arange(1, D + 1, 1))))
    return lognorm


def normalwishart_grad_log_norm(natural_params):
    np1, np2, np3, np4, D = normalwishart_split_np(natural_params)

    outer = np.outer(np2, np2) / np3
    matrix = (np1 - outer)
    sign, logdet = np.linalg.slogdet(matrix)
    inv_matrix = np.linalg.inv(matrix)

    grad1 = -.5 * (np4 + D) * inv_matrix
    grad2 = (np4 + D) * inv_matrix @ (np2 / np3)
    grad3 = - D / (2 * np3) - .5 * (np4 + D) \
        * np.trace(inv_matrix @ (outer / np3))
    grad4 = .5 * np.sum(psi(.5 * (np4 + D + 1 - np.arange(1, D + 1, 1))))
    grad4 += -.5 * sign * logdet + .5 * D * np.log(2)
    return np.hstack([grad1.reshape(-1), grad2, grad3, grad4])


class TestDirichlet(unittest.TestCase):

    def test_create(self):
        model = beer.dirichlet(torch.ones(4))
        self.assertTrue(isinstance(model, beer.ExpFamilyDensity))
        self.assertAlmostEqual(model.natural_params.data.sum(), 0.)

    def test_exp_sufficient_statistics(self):
        model = beer.dirichlet(torch.ones(4))
        model_s_stats = model.expected_sufficient_statistics.data.numpy()
        natural_params = model.natural_params.data.numpy()
        s_stats = dirichlet_grad_log_norm(natural_params)
        self.assertTrue(np.allclose(model_s_stats, s_stats))

    def test_kl_divergence(self):
        model1 = beer.dirichlet(torch.ones(4))
        model2 = beer.dirichlet(torch.ones(4))
        div = beer.kl_divergence(model1, model2)
        self.assertAlmostEqual(div, 0.)

    def test_log_norm(self):
        model = beer.dirichlet(torch.ones(4))
        model_log_norm = model.log_norm.data.numpy()
        natural_params = model.natural_params.data.numpy()
        log_norm = dirichlet_log_norm(natural_params)
        self.assertAlmostEqual(model_log_norm, log_norm)


class TestNormalGamma(unittest.TestCase):

    def test_create(self):
        model = beer.normalgamma(torch.zeros(2), torch.ones(2), 1.)
        self.assertTrue(isinstance(model, beer.ExpFamilyDensity))

    def test_exp_sufficient_statistics(self):
        model = beer.normalgamma(torch.zeros(2), torch.ones(2), 1.)
        model_s_stats = model.expected_sufficient_statistics.data.numpy()
        natural_params = model.natural_params.data.numpy()
        s_stats = normalgamma_grad_log_norm(natural_params)
        self.assertTrue(np.allclose(model_s_stats, s_stats))

    def test_kl_divergence(self):
        model1 = beer.normalgamma(torch.zeros(2), torch.ones(2), 1.)
        model2 = beer.normalgamma(torch.zeros(2), torch.ones(2), 1.)
        div = beer.kl_divergence(model1, model2)
        self.assertAlmostEqual(div, 0.)

    def test_log_norm(self):
        model = beer.normalgamma(torch.zeros(2), torch.ones(2), 1.)
        model_log_norm = model.log_norm.data.numpy()
        natural_params = model.natural_params.data.numpy()
        log_norm = normalgamma_log_norm(natural_params)
        self.assertTrue(len(model_log_norm) == 1 and model_log_norm.shape[0] == 1)
        self.assertAlmostEqual(model_log_norm[0], log_norm)


class TestNormalWishart(unittest.TestCase):

    def test_create(self):
        model = beer.normalwishart(torch.zeros(2), torch.eye(2), 1.)
        self.assertTrue(isinstance(model, beer.ExpFamilyDensity))

    def test_exp_sufficient_statistics(self):

        model = beer.normalwishart(torch.zeros(2), torch.eye(2), 1.)
        model_s_stats = model.expected_sufficient_statistics.data.numpy()
        natural_params = model.natural_params.data.numpy()
        s_stats = normalwishart_grad_log_norm(natural_params)
        print(model_s_stats, s_stats)
        self.assertTrue(np.allclose(model_s_stats, s_stats))

    def test_kl_divergence(self):
        model1 = beer.normalwishart(torch.zeros(2), torch.eye(2), 1.)
        model2 = beer.normalwishart(torch.zeros(2), torch.eye(2), 1.)
        div = beer.kl_divergence(model1, model2)
        self.assertAlmostEqual(div, 0.)

    def test_log_norm(self):
        model = beer.normalwishart(torch.zeros(2), torch.eye(2), 1.)
        model_log_norm = model.log_norm.data.numpy()
        natural_params = model.natural_params.data.numpy()
        log_norm = normalwishart_log_norm(natural_params)
        self.assertTrue(len(model_log_norm) == 1 and model_log_norm.shape[0] == 1)
        self.assertAlmostEqual(model_log_norm[0], log_norm, places=4)


if __name__ == '__main__':
    unittest.main()
