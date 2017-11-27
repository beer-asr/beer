
"""Generate synthetic data to test the Bayesian PPCA model."""

import argparse
import numpy as np


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsamples', type=int, default=100,
                        help='number of samples to generate (100)')
    parser.add_argument('--dimension', type=int, default=2,
                        help='dimension of the data (2)')
    parser.add_argument('--subspace-dimension', type=int, default=1,
                        help='underlying dimension of the subspace.')
    parser.add_argument('--subspace-variance', type=float, default=1.,
                        help='variance for each dimension of the underlying '
                             'subspace (1.)')
    parser.add_argument('--noise-variance', type=float, default=.1,
                        help='noise variance (0.1)')
    parser.add_argument('out', help='output filename (without "npy" '
                                    'extension)')
    args = parser.parse_args()

    mean = np.random.randn(args.dimension)
    cov = np.diag(np.ones(args.dimension) * args.noise_variance)
    idxs = range(args.subspace_dimension)
    cov[idxs, idxs] = args.subspace_variance
    data = np.random.multivariate_normal(mean, cov,
        size=args.nsamples).T

    # Generate random orthonormal bases.
    W, _ = np.linalg.qr(np.random.randn(args.dimension, args.dimension))

    # Project the data onto theses bases.
    projected_data = W @ data

    # Save the data on disk.
    np.save(args.out, projected_data)

if __name__ == '__main__':
    run()
