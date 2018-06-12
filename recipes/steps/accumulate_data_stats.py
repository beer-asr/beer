
'''Accumulate data statistics: mean, standard deviation and total data points
'''

import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=str, help='Feature file')
    args = parser.parse_args()

    data_file = args.features
    data_dir = os.path.dirname(os.path.abspath(data_file))
    data = np.load(data_file)
    keys = list(data.keys())
    dim = data[data.keys()[0]].shape[1]
    data_sum = np.zeros(dim)
    square_sum = np.zeros(dim)
    count = 0
    for k in keys:
        data_sum += data[k].sum(axis=0)
        square_sum += (data[k] ** 2).sum(axis=0)
        count += data[k].shape[0]
    mean = data_sum / count
    std = np.sqrt(square_sum / count - mean ** 2)
    stats = {'mean': mean, 'std': std, 'count': count}
    np.savez(data_dir + '/feats_stats.npz', **stats)

if __name__ == "__main__":
    main()
