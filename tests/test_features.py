'Test the features module.'

# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import beer
import numpy as np
from basetest import BaseTest


class TestFbank(BaseTest):

    def test_fbank(self):
        ref_fea = np.load('tests/fbank.npy')
        s_t = np.load('tests/audio.npy')
        fea = beer.features.fbank(s_t, nfilters=30, lowfreq=100)
        self.assertTrue(np.allclose(ref_fea, fea))

    def test_deltas(self):
        ref_fea = np.load('tests/fbank_d_dd.npy')
        s_t = np.load('tests/audio.npy')
        fea = beer.features.fbank(s_t, nfilters=30, lowfreq=100)
        fea_d_dd = beer.features.add_deltas(fea)
        self.assertTrue(np.allclose(ref_fea, fea_d_dd))


__all__ = ['TestFbank']
