
'Re-estimate a unigram LM.'

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch

import beer


log_format = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lm', help='unigram language model to train')
    parser.add_argument('data', help='data')
    parser.add_argument('outlm', help='output model')
    args = parser.parse_args()

    # Load the model.
    with open(args.lm, 'rb') as fh:
        model = pickle.load(fh)

    # Load the data for the training.
    data = np.load(args.data)

    # Count the number of in the training data.
    tot_counts = 0
    for utt in data:
        tot_counts += len(data[utt])

    # Prepare the optimizer for the training.
    params = model.mean_field_groups
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(params, lrate=1.)
    optimizer.zero_grad()

    # Initialize the objective function.
    elbo = beer.evidence_lower_bound(datasize=tot_counts)

    # Re-estimate the LM.
    for utt in data:
        ft = torch.from_numpy(data[utt])
        elbo += beer.evidence_lower_bound(model, ft, datasize=tot_counts)
    elbo.natural_backward()
    optimizer.step()

    # Save the model.
    with open(args.outlm, 'wb') as fh:
        model = pickle.dump(model, fh)


if __name__ == "__main__":
    main()
