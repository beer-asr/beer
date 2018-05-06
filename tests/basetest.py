
''' Base class for the test cases.

Python ``unittest`` framework does not allows large scale of
parameterized tests. This simple base class is just a workaround
to allow to run the tests for various conditions.

'''


import numpy as np
import torch
import unittest


class BaseTest(unittest.TestCase):

    def __init__(self, methodName='runTest', tensor_type='float', gpu=False,
                 seed=13):
        super().__init__(methodName)
        self.tensor_type = tensor_type
        self.gpu = gpu
        self.seed(seed)

    def seed(self, seed):
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed(seed)

    def assertArraysAlmostEqual(self, arr1, arr2):
        self.assertTrue(np.allclose(arr1, arr2))

    @staticmethod
    def get_testsuite(class_name, tensor_type='float', gpu=False, seed=13):
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(class_name)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(class_name(name, tensor_type, gpu, seed))
        return suite
