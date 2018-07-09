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

    def test_load_value(self):
        self.assertEqual(1, beer.models.nnet.load_value('1'))
        self.assertTrue(beer.models.nnet.load_value('True'))
        self.assertAlmostEqual(1.3, beer.models.nnet.load_value('1.3'))
        self.assertEqual(2, beer.models.nnet.load_value('%var1 + %var2',
                                                        {'%var1': 1,
                                                         '%var2': 1}))

    def test_parse_nnet_element(self):
        strval = 'Linear:in_features=10,out_features=20'
        fn_name, str_kwargs = beer.models.nnet.parse_nnet_element(strval)
        self.assertEqual(fn_name, 'Linear')
        self.assertEqual(str_kwargs['in_features'], '10')
        self.assertEqual(str_kwargs['out_features'], '20')

        strval = 'Tanh'
        fn_name, str_kwargs = beer.models.nnet.parse_nnet_element(strval)
        self.assertEqual(fn_name, 'Tanh')

    def test_create_nnet_element(self):
        strval = 'Linear:in_features=10,out_features=20'
        linear_layer = beer.models.nnet.create_nnet_element(strval)
        self.assertTrue(isinstance(linear_layer, torch.nn.Linear))
        self.assertEqual(linear_layer.in_features, 10)
        self.assertEqual(linear_layer.out_features, 20)

        strval = 'Tanh'
        tanh = beer.models.nnet.create_nnet_element(strval)
        self.assertTrue(isinstance(tanh, torch.nn.Tanh))

        strval = 'ELU:alpha=.5,inplace=True'
        elu = beer.models.nnet.create_nnet_element(strval)
        self.assertTrue(isinstance(elu, torch.nn.ELU))
        self.assertTrue(elu.inplace)
        self.assertAlmostEqual(elu.alpha, .5)

    def test_create_nnet_block(self):
        variables = {'%feadim': self.dim}
        block_conf = {
            'structure': [
                'Linear:in_features=%feadim,out_features=20',
                'Tanh',
                'Linear:in_features=20,out_features=%feadim',
                'Sigmoid'
            ],
            'residual_connection': 'none'
        }
        nnet_block = beer.models.nnet.create_nnet_block(block_conf, variables)
        nnet_block = nnet_block.type(self.type)
        nnet_block(self.data)

        block_conf['residual_connection'] = 'none'
        nnet_block = beer.models.nnet.create_nnet_block(block_conf, variables)
        nnet_block = nnet_block.type(self.type)
        nnet_block(self.data)

        block_conf['residual_connection'] = 'identity'
        nnet_block = beer.models.nnet.create_nnet_block(block_conf, variables)
        nnet_block = nnet_block.type(self.type)
        nnet_block(self.data)

        block_conf = {
            'structure': [
                'Linear:in_features=%feadim,out_features=20',
                'Tanh',
                'Linear:in_features=20,out_features=20',
                'Sigmoid'
            ],
            'residual_connection': 'Linear:in_features=%feadim,out_features=20'
        }
        nnet_block = beer.models.nnet.create_nnet_block(block_conf, variables)
        nnet_block = nnet_block.type(self.type)
        nnet_block(self.data)

    def test_create_encoder_nnet(self):
        variables = {'%feadim': self.dim}
        conf = {
            'blocks': [
                {
                    'structure': [
                        'Linear:in_features=%feadim,out_features=20',
                        'Tanh',
                        'Linear:in_features=20,out_features=%feadim',
                        'Sigmoid'
                    ],
                    'residual_connection': 'identity'
                },
                {
                    'structure': [
                        'Linear:in_features=%feadim,out_features=20',
                        'Sigmoid',
                        'Linear:in_features=20,out_features=%feadim',
                        'Sigmoid'
                    ],
                    'residual_connection': 'none'
                }
            ],
            'dim_input_normal_layer': '%feadim',
            'dim_output_normal_layer': '30',
            'covariance': 'diagonal'
        }
        encoder = beer.models.nnet.create_encoder(conf, variables)
        encoder = encoder.type(self.type)
        outputs = encoder(self.data)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape[1], 30)
        self.assertEqual(outputs[1].shape[1], 30)

        conf['covariance'] = 'isotropic'
        encoder = beer.models.nnet.create_encoder(conf, variables)
        encoder = encoder.type(self.type)
        outputs = encoder(self.data)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape[1], 30)
        self.assertEqual(outputs[1].shape[1], 30)


    def test_create_decoder_nnet(self):
        variables = {'%feadim': self.dim}
        conf = {
            'blocks': [
                {
                    'structure': [
                        'Linear:in_features=%feadim,out_features=20',
                        'Tanh',
                        'Linear:in_features=20,out_features=30',
                        'Sigmoid'
                    ],
                    'residual_connection': 'none'
                },
                {
                    'structure': [
                        'Linear:in_features=30,out_features=20',
                        'Sigmoid',
                        'Linear:in_features=20,out_features=30',
                        'Sigmoid'
                    ],
                    'residual_connection': 'none'
                }
            ],
            'dim_input_normal_layer': '30',
            'dim_output_normal_layer': '%feadim',
        }
        decoder = beer.models.nnet.create_decoder(conf, variables)
        decoder = decoder.type(self.type)
        outputs = decoder(self.data)
        self.assertEqual(outputs.shape[1], self.dim)


__all__ = [
    'TestNeuralNetwork',
]
