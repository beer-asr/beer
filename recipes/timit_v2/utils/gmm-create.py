'Create a the GMM.'

import argparse
import pickle
import torch
import yaml

import numpy as np
import beer



def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stats', help='Feature statistics file for gmm model')
    group.add_argument('--dim', type=int,
                        help='Dimension of feature, used for gmm model')
    parser.add_argument('conf', help='Configuration file')
    parser.add_argument('n_normal', type=int, help='Number of Gaussian pdfs')
    parser.add_argument('gmm', help='outout gmm')
    args = parser.parse_args()

    # Load the GMM configuration.
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

    modelset = beer.NormalSet.create(
        mean, var,
        size=args.n_normal,
        prior_strength=conf['prior_strength'],
        noise_std=conf['noise_std'],
        cov_type=conf['cov_type'],
        shared_cov=conf['shared_cov']
    )
    gmm = beer.Mixture.create(modelset, prior_strength=conf['prior_strength'])

    with open(args.gmm, 'wb') as fid:
        pickle.dump(gmm, fid)


if __name__ == '__main__':
    main()
