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

    def test_log_norm(self):
        model = beer.dirichlet(torch.ones(4))
        model_log_norm = model.log_norm.data.numpy()
        natural_params = model.natural_params.data.numpy()
        log_norm = dirichlet_log_norm(natural_params)
        self.assertAlmostEqual(model_log_norm, log_norm)


if __name__ == '__main__':
    unittest.main()
