
'''Compute statisttics (mean, var, count) of a database.'''

import argparse
import copy
import os
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import beer


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('list', help='list of npz archives')
    parser.add_argument('out', help='output path for the statistics')
    args = parser.parse_args()

    with open(args.list, 'r') as fid:
        for i, line in enumerate(fid):
            data = np.load(line.strip())['features']
            if i == 0:
                acc_x = np.sum(data, axis=0)
                acc_x2 = np.sum(data**2, axis=0)
                counts = len(data)
            else:
                acc_x += np.sum(data, axis=0)
                acc_x2 += np.sum(data**2, axis=0)
                counts += len(data)

    mean = acc_x / counts
    variance = (acc_x2 / counts) - mean ** 2
    np.savez(args.out, mean=mean, variance=variance,
             counts=np.asarray(counts))


if __name__ == '__main__':
    run()
0
