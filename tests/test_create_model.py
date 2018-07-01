'Test the Normal model.'

import yaml
import sys
sys.path.insert(0, './')
import os
import glob
import unittest
import torch
import beer
from basetest import BaseTest


model_confs = [

]


class TestCreateModel(BaseTest):

    def setUp(self):
        self.conf_files = []
        for path in glob.glob('./tests/models/*yml'):
            self.conf_files.append(path)
        assert len(self.conf_files) > 0

        self.dim = int(10 + torch.randint(100, (1, 1)).item())
        self.mean = torch.randn(self.dim).type(self.type)
        self.variance = (1 + torch.randn(self.dim) ** 2).type(self.type)

    def test_create_model_uknown_type(self):
        test_yaml = {
            'type': 'UnknownModel',
            'property1': 'abc',
            'property2': 'abc'
        }
        self.assertRaises(ValueError, beer.create_model, test_yaml,
                          self.mean, self.variance)

    def test_create_model(self):
        for conf_file in self.conf_files:
            with self.subTest(conf_file=conf_file):
                # Doesn't test the content of the object, just make sure
                # that the creation does not crash and return a valid
                # object.
                with open(conf_file, 'r') as fid:
                    conf = yaml.load(fid)
                    model = beer.create_model(conf, self.mean, self.variance)
                self.assertTrue(isinstance(model, beer.BayesianModel))


__all__ = ['TestCreateModel']
