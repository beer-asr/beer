'Test the nnet module.'

import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')
import glob
import yaml
import numpy as np
import torch

import beer
from basetest import BaseTest


class TestNeuralNetwork(BaseTest):

    def test_load_value(self):
        self.assertEqual(1, beer.models.nnet.load_value('1'))
        self.assertTrue(beer.models.nnet.load_value('True'))
        self.assertAlmostEqual(1.3, beer.models.nnet.load_value('1.3'))
        self.assertEqual(2, beer.models.nnet.load_value('%var1 + %var2',
                                                        {'%var1': 1,
                                                         '%var2': 1}))



__all__ = [
    'TestNeuralNetwork',
]
