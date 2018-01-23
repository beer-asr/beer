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


def dirichlet_sufficient_statistics(X):
    # (Invalid Argument Name) pylint: disable=C0103
    return np.log(X)


class TestDirichlet(unittest.TestCase):

    def test_create_from_tensor(self):
        model = beer.dirichlet(torch.ones(4))
        self.assertTrue(isinstance(model, beer.ExpFamilyDensity))
        self.assertAlmostEqual(model.natural_params.data.sum(), 0.)

    def test_create_from_variable(self):
        model = beer.dirichlet(Variable(torch.ones(4), requires_grad=True))
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

    def test_log_base_measure(self):
        model = beer.dirichlet(torch.ones(4))
        X = torch.from_numpy(np.array([[0.1, 0.9], [0.5, 0.5], [0.3, 0.7]]))
        log_bmeasure = model.log_base_measure(X)
        self.assertAlmostEqual(log_bmeasure, 0.)

    def test_log_likelihood(self):
        model = beer.dirichlet(torch.ones(2).double())
        X = Variable(torch.from_numpy(np.array([
            [0.1, 0.9],
            [0.5, 0.5],
            [0.3, 0.7]
        ])))
        s_stats = dirichlet_sufficient_statistics(X.numpy())
        natural_params = model.natural_params.data.numpy()
        log_norm = dirichlet_log_norm(natural_params)
        llh = s_stats @ natural_params - log_norm
        model_llh = model.log_likelihood(X).data.numpy()
        self.assertTrue(np.allclose(model_llh, llh))

    def test_log_norm(self):
        model = beer.dirichlet(torch.ones(4))
        model_log_norm = model.log_norm.data.numpy()
        natural_params = model.natural_params.data.numpy()
        log_norm = dirichlet_log_norm(natural_params)
        self.assertAlmostEqual(model_log_norm, log_norm)

    def test_sufficient_statistics(self):
        model = beer.dirichlet(torch.ones(4))
        X = torch.from_numpy(np.array([[0.1, 0.9], [0.5, 0.5], [0.3, 0.7]]))
        s_stats = dirichlet_sufficient_statistics(X.numpy())
        model_s_stats = model.sufficient_statistics(X).numpy()
        self.assertTrue(np.allclose(model_s_stats, s_stats))


if __name__ == '__main__':
    unittest.main()
