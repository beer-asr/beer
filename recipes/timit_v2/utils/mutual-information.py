'''Estimate the mutual information.

This script is to be used with "utils/compare-alignments.py"

'''

import argparse
from collections import defaultdict
import logging
import math
import pickle

import numpy as np

logging.basicConfig(format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', action='store_true',
                        help='show debug messages')
    parser.add_argument('data', help='normalized confusion and marginals')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    with open(args.data, 'rb') as f:
        joint_counts, ref_counts, hyp_counts = pickle.load(f)

    # Normalize the count to get a valid distribution.
    total = np.sum([count for count in ref_counts.values()])
    for ref_sym in ref_counts:
        ref_counts[ref_sym] /= total
    total = np.sum([count for count in hyp_counts.values()])
    for hyp_sym in hyp_counts:
        hyp_counts[hyp_sym] /= total
    for ref_sym in ref_counts:
        total = np.sum([count for count in joint_counts[ref_sym].values()])
        for hyp_sym in joint_counts[ref_sym]:
            joint_counts[ref_sym][hyp_sym] /= total


    p_X_given_Y = joint_counts
    p_X = hyp_counts
    p_Y = ref_counts

    H_X = np.sum([-prob * np.log(prob) for prob in p_X.values()]) / math.log(2)
    H_Y = np.sum([-prob * np.log(prob) for prob in p_Y.values()]) / math.log(2)
    H_X_given_Y = np.sum([
        prob_y * np.sum([-(prob_x) * np.log(prob_x)
                           for prob_x in p_X_given_Y[y].values()])
        for y, prob_y in p_Y.items()
    ]) / math.log(2)

    logging.debug(f'H(X) = {H_X:.2f}  H(Y) = {H_Y:.2f} H(X|Y) = {H_X_given_Y:.2f}')
    print(f'MI(X, Y) = {H_X - H_X_given_Y:.2f} (bits)')
    print(f'norm. MI(X, Y) = {100 * (H_X - H_X_given_Y) / H_Y:.2f} %')


if __name__ == "__main__":
    main()
