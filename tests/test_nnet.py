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

    def setUp(self):
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)

        self.conf_files = []
        for path in glob.glob('./tests/nnets/*yml'):
            self.conf_files.append(path)
        assert len(self.conf_files) > 0

    def test_load_value(self):
        self.assertEqual(1, beer.nnet.neuralnetwork.load_value('1'))
        self.assertTrue(beer.nnet.neuralnetwork.load_value('True'))
        self.assertAlmostEqual(1.3, beer.nnet.neuralnetwork.load_value('1.3'))
        self.assertEqual(2, beer.nnet.neuralnetwork.load_value('%var1 + %var2',
                                                        {'%var1': 1,
                                                         '%var2': 1}))

    def test_parse_nnet_element(self):
        strval = 'Linear:in_features=10;out_features=20'
        fn_name, str_kwargs = beer.nnet.neuralnetwork.parse_nnet_element(strval)
        self.assertEqual(fn_name, 'Linear')
        self.assertEqual(str_kwargs['in_features'], '10')
        self.assertEqual(str_kwargs['out_features'], '20')

        strval = 'Tanh'
        fn_name, str_kwargs = beer.nnet.neuralnetwork.parse_nnet_element(strval)
        self.assertEqual(fn_name, 'Tanh')

    def test_create_nnet_element(self):
        strval = 'Linear:in_features=10;out_features=20'
        linear_layer = beer.nnet.neuralnetwork.create_nnet_element(strval)
        self.assertTrue(isinstance(linear_layer, torch.nn.Linear))
        self.assertEqual(linear_layer.in_features, 10)
        self.assertEqual(linear_layer.out_features, 20)

        strval = 'Tanh'
        tanh = beer.nnet.neuralnetwork.create_nnet_element(strval)
        self.assertTrue(isinstance(tanh, torch.nn.Tanh))

        strval = 'ELU:alpha=.5;inplace=True'
        elu = beer.nnet.neuralnetwork.create_nnet_element(strval)
        self.assertTrue(isinstance(elu, torch.nn.ELU))
        self.assertTrue(elu.inplace)
        self.assertAlmostEqual(elu.alpha, .5)

    def test_ReshapeLayer(self):
        strval = 'ReshapeLayer:shape=(-1,10,20)'
        reshape_layer = beer.nnet.neuralnetwork.create_nnet_element(strval)
        self.assertTrue(isinstance(reshape_layer, beer.nnet.neuralnetwork.ReshapeLayer))
        self.assertEqual(reshape_layer.shape[0], -1)
        self.assertEqual(reshape_layer.shape[1], 10)
        self.assertEqual(reshape_layer.shape[2], 20)

        input_data = torch.randn(100, 200).type(self.type)
        output_data = reshape_layer(input_data)
        self.assertEqual(output_data.shape[0], 100)
        self.assertEqual(output_data.shape[1], 10)
        self.assertEqual(output_data.shape[2], 20)

    def test_nnet_forward(self):
        dtype, device = self.data.dtype, self.data.device
        for conf_file in self.conf_files:
            with self.subTest(conf_file=conf_file):
                # Doesn't test the content of the object, just make sure
                # that the creation does not crash.
                with open(conf_file, 'r') as fid:
                    conf = yaml.load(fid)
                    nnet = beer.nnet.neuralnetwork.create(conf, dtype, device,
                                                          {'<feadim>': self.dim})
                    nnet(self.data)

__all__ = [
    'TestNeuralNetwork',
]
