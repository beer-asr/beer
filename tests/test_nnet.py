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

    def test_parse_nnet_element(self):
        strval = 'Linear:in_features=10,out_features=20'
        fn, kwargs = beer.models.nnet.parse_nnet_element(strval)
        linear_layer = fn(**kwargs)
        self.assertTrue(isinstance(linear_layer, torch.nn.Linear))
        self.assertEqual(linear_layer.in_features, 10)
        self.assertEqual(linear_layer.out_features, 20)

        strval = 'Tanh'
        fn, kwargs = beer.models.nnet.parse_nnet_element(strval)
        tanh = fn(**kwargs)
        self.assertTrue(isinstance(tanh, torch.nn.Tanh))

        strval = 'ELU:alpha=.5,inplace=True'
        fn, kwargs = beer.models.nnet.parse_nnet_element(strval)
        elu = fn(**kwargs)
        self.assertTrue(isinstance(elu, torch.nn.ELU))
        self.assertTrue(elu.inplace)
        self.assertAlmostEqual(elu.alpha, .5)

__all__ = [
    'TestNeuralNetwork',
]
