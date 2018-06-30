'Test the Normal model.'

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
        self.confs = []
        for path in glob.glob('./tests/models/*yml'):
            self.confs.append(path)

        assert len(self.confs) > 0

    def test_create_model_uknown_type(self):
        test_yaml = """
        type: UnknownModel
        property1: abc
        property2: abc
        """
        self.assertRaises(ValueError, beer.create_model, test_yaml)

    def test_create_model(self):
        for i, conf in enumerate(self.confs):
            with self.subTest(i=i):
                # Doesn't test the content of the object, just make sure
                # that the creation does not crash and return a valid
                # object.
                with open(conf, 'r') as fid:
                    model = beer.create_model(fid)
                self.assertTrue(isinstance(model, beer.BayesianModel))


__all__ = ['TestCreateModel']
