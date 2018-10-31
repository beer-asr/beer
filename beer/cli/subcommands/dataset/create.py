
'compile a data set with the given features'

import argparse
import os
import pickle

import numpy as np
import torch

from ...dataset import Dataset


def accumulate(feature_file):
    '''Compute global mean, variance, frame counts
    Argument:
        feature_file(str): feature file(npz)
    Returns:
        mean: np array (float)
        var: np array (float)
        tot_counts(int): total frames in feature files
    '''
    feats = np.load(feature_file)
    keys = list(feats.keys())
    dim = feats[keys[0]].shape[1]
    tot_sum = np.zeros(dim)
    tot_square_sum = np.zeros(dim)
    tot_counts = 0
    for k in keys:
        nframes_per_utt = len(feats[k])
        per_square_sum = (feats[k] ** 2).sum(axis=0)
        tot_sum += feats[k].sum(axis=0)
        tot_square_sum += per_square_sum
        tot_counts += nframes_per_utt
    mean = tot_sum / tot_counts
    var = tot_square_sum / tot_counts - mean ** 2
    return torch.from_numpy(mean).float(), torch.from_numpy(var).float(), \
           int(tot_counts)


def setup(parser):
    parser.add_argument('datadir', help='data directory')
    parser.add_argument('features', help='features archive (npz format)')
    parser.add_argument('out', help='output compiled dataset')


def main(args, logger):
    logger.debug('computing features statistics...')
    mean, var, size = accumulate(args.features)

    logger.debug('creating the dataset...')
    dataset = Dataset(os.path.abspath(args.features), mean, var, size)

    logger.debug('saving the dataset on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(dataset, f)

    logger.info(f'created dataset with {len(dataset)} utterances '\
                f'(total frame count: {dataset.size})')

if __name__ == "__main__":
    main()

