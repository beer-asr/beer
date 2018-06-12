
'''Create emission models
'''

import numpy as np
import argparse
import sys
sys.path.insert(0, '../../beer')
import beer
import pickle
import os
import torch

emission_types = ['norm_diag', 'norm_full', 'norm_shared_diag',
                  'norm_shared_full']
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nstates', type=int, help='state number')
    parser.add_argument('stats', type=str, help='data statistics')
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--emission_type', default='norm_diag',
        choices=emission_types)
    parser.add_argument('--mean_normalize', default='True',
        choices=['True', 'False'])
    parser.add_argument('--var_normalize', default='True',
        choices=['True', 'False'])
    args = parser.parse_args()

    nstates = args.nstates
    stats = np.load(args.stats)
    modeltype = args.emission_type
    mdl = args.model_dir + '/emission.mdl'

    models = [beer.NormalDiagonalCovarianceSet,
              beer.NormalFullCovarianceSet,
              beer.NormalSetSharedDiagonalCovariance,
              beer.NormalSetSharedFullCovariance]
    dict_models = dict((k, v) for (k, v) in zip(emission_types, models))
    model = dict_models[modeltype]

    mean = torch.from_numpy(stats['mean']).float()
    std = torch.from_numpy(stats['std']).float()

    if args.mean_normalize == 'True':
        mean = torch.zeros_like(mean)
    if args.var_normalize == 'True':
        std = torch.ones_like(var)

    modelset = model.create(mean, std, nstates, noise_std=0.)

    with open(mdl, 'wb') as m:
        pickle.dump(modelset, m)

if __name__ == '__main__':
    main()
