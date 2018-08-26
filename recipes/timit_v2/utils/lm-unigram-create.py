
'Create a Bayesian unigram LM.'

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
    parser.add_argument('--concentration', type=float, default=1,
                        help='Concentration of the dirichlet prior')
    parser.add_argument('vocsize', type=int, help='size of the vocabulary')
    parser.add_argument('outlm', help='output model')
    args = parser.parse_args()

    model = beer.UnigramLM.create(args.vocsize,
                                  prior_strength=args.concentration)

    with open(args.outlm, 'wb') as fh:
        pickle.dump(model, fh)


if __name__ == "__main__":
    main()
