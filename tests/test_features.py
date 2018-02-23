'Test the features module.'


import sys
sys.path.insert(0, './')
import unittest
import beer
import numpy as np
from scipy.io.wavfile import read


class TestFbank(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()

