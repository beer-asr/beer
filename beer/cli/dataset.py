'Structure over a dataset.'

from dataclasses import dataclass, field
import random
from typing import NamedTuple, Any
import numpy as np
import torch


class Utterance(NamedTuple):
    'An audio recording and the associated meta-data.'
    id: str
    features: torch.Tensor
    #speaker: str = None
    #transcription: str = None


class UtteranceIterator:

    def __init__(self, utts, fea_dict):
        self.utts = utts
        self.fea_dict = fea_dict
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            uttid = self.utts[self.idx]
        except IndexError:
            raise StopIteration
        features = torch.from_numpy(self.fea_dict[uttid]).float()
        self.idx += 1
        return Utterance(id=uttid, features=features)


@dataclass
class Dataset:
    'A collection of utterances with their features and meta-data.'

    feapath: str
    mean: torch.Tensor
    var: torch.Tensor
    size: int
    _fea_dict: Any = field(default=None)

    @property
    def fea_dict(self):
        if self._fea_dict is None:
            self._fea_dict = np.load(self.feapath)
        return self._fea_dict

    def __getstate__(self):
        self._fea_dict = None
        return self.__dict__

    def __len__(self):
        features = np.load(self.feapath)
        return len(features.files)

    def utterances(self, random_order=True):
        '''Return an iterator over the utterances.

        Args:
            random_order (boolean): If False, iterate over the
                utterances sorted alphabetically by their id.

        Returns:
            ``iterable``

        '''
        uttsid = sorted(list(self.fea_dict.keys()))
        if random_order:
            random.shuffle(uttsid)
        return UtteranceIterator(uttsid, self.fea_dict)

    def __getitem__(self, key):
        features = torch.from_numpy(self.fea_dict[key]).float()
        return Utterance(key, features)

