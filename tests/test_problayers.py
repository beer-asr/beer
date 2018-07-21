'Test the nnet.problayers module.'

import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')
import glob
import yaml
import numpy as np
import torch

import beer
from basetest import BaseTest


class TestProbabilisticLayers(BaseTest):

    def setUp(self):
        self.conf_files = []
        for path in glob.glob('./tests/problayers/*yml'):
            self.conf_files.append(path)
        assert len(self.conf_files) > 0

    def test_create_problayer(self):
        for conf_file in self.conf_files:
            with self.subTest(conf_file=conf_file):
                # Doesn't test the content of the object, just make sure
                # that the creation does not crash.
                with open(conf_file, 'r') as fid:
                    conf = yaml.load(fid)
                    layer = beer.nnet.problayers.create(conf)
                self.assertIsNotNone(layer)

__all__ = [
    'TestProbabilisticLayers',
]
