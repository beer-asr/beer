
'Train a HMM with a given alignments.'

import random
from collections import defaultdict
from functools import partial
import numpy as np
import torch
import argparse
import sys
import beer
import pickle
import logging
import os


log_format = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--epsilon', type=float, default=1e-12,
                        help='small constant to add to the confusion ' \
                             'matrix to avoid overflow')
    parser.add_argument('model', help='model to train')
    parser.add_argument('batches', help='list of batches file')
    #parser.add_argument('feat_stats', help='data statistics')
    parser.add_argument('out', help='output')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load the batches.
    batches_list = []
    with open(args.batches, 'r') as f:
        for line in f:
            batches_list.append(line.strip())

    # Load the model and move it to the chosen device (CPU/GPU)
    with open(args.model, 'rb') as fh:
        model = pickle.load(fh)
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    joint_counts = defaultdict(partial(defaultdict, partial(float, args.epsilon)))
    ref_counts = defaultdict(partial(float, args.epsilon))
    hyp_counts = defaultdict(partial(float, args.epsilon))
    for batch_no, path in enumerate(batches_list, start=1):
        # Load the batch data.
        batch = np.load(path)
        ft = torch.from_numpy(batch['features']).float()
        labels = torch.from_numpy(batch['labels']).long()
        labels = labels.to(device)
        ft = ft.to(device)

        #import pdb; pdb.set_trace()
        pred = model.posteriors(ft).argmax(dim=1)

        for i, frame in enumerate(pred):
            l = int(labels[i])
            p = int(pred[i])
            joint_counts[l][p] += 1
            ref_counts[l] += 1
            hyp_counts[p] += 1

    with open(args.out, 'wb') as f:
        pickle.dump((joint_counts, ref_counts, hyp_counts), f)


if __name__ == "__main__":
    main()
