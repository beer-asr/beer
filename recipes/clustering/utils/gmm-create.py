'Create a the GMM.'

import numpy as np
import argparse
import beer
import pickle
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stats', help='feature statistics to ' \
                                       'initialize the model')
    group.add_argument('--dim', type=int,
                        help='dimension of feature')
    parser.add_argument('conf', help='configuration file')
    parser.add_argument('out', help='outout emissions')
    args = parser.parse_args()

    # Load the model configuration.
    with open(args.conf, 'r') as fid:
        conf = yaml.load(fid)

    # Get the data statistics.
    if args.stats:
        stats = np.load(args.stats)
        mean = torch.from_numpy(stats['mean']).float()
        var = torch.from_numpy(stats['var']).float()
    else:
        dim = args.dim
        mean = torch.zeros(dim).float()
        var = torch.ones(dim).float()


    modelset = beer.NormalSet.create(mean, var, size=conf['size'],
                                     prior_strength=conf['prior_strength'],
                                     noise_std=conf['noise_std'],
                                     cov_type=conf['cov_type'],
                                     shared_cov=conf['shared_cov'])
    model = beer.Mixture.create(modelset,
                                prior_strength=conf['concentration'])

    with open(args.out, 'wb') as fid:
        pickle.dump(model, fid)

if __name__ == '__main__':
    main()

