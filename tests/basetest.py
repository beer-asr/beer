
''' Base class for the test cases.

Python ``unittest`` framework does not allows large scale of
parameterized tests. This simple base class is just a workaround
to allow to run the tests for various conditions.

'''


import numpy as np
import torch
import unittest


FLOAT_TOLPLACES = 1
FLOAT_TOL = 10 ** (-FLOAT_TOLPLACES)

DOUBLE_TOLPLACES = 6
DOUBLE_TOL = 10 ** (-DOUBLE_TOLPLACES)


class BaseTest(unittest.TestCase):

    def __init__(self, methodName='runTest', tensor_type='float', gpu=False,
                 seed=13):
        super().__init__(methodName)
        self.tensor_type = tensor_type
        self.gpu = gpu
        self.seed(seed)

    @property
    def tol(self):
        return FLOAT_TOL if self.tensor_type == 'float' else DOUBLE_TOL

    @property
    def tolplaces(self):
        return FLOAT_TOLPLACES if self.tensor_type == 'float' else \
            DOUBLE_TOLPLACES

    @property
    def type(self):
        return torch.FloatTensor if self.tensor_type == 'float' else \
            torch.DoubleTensor

    def seed(self, seed):
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed(seed)

    def assertArraysAlmostEqual(self, arr1, arr2):
        try:
            self.assertTrue(np.allclose(arr1, arr2, atol=self.tol))
            fail = False
        except AssertionError as error:
            fail = True
            raise error
        finally:
            if fail:
                print(arr1, arr2)



    @staticmethod
    def get_testsuite(class_name, tensor_type='float', gpu=False, seed=13):
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(class_name)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(class_name(name, tensor_type, gpu, seed))
        return suite
