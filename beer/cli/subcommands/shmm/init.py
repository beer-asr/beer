'initialize the phone loop depending on the GSM'

import argparse
import copy
import math
import os
import pickle
import sys

import torch

import beer


# Create a view of the emissions (aka modelset) for each units.
def iterate_units(modelset, nunits, nstates):
    for idx in range(nunits):
        start, end = idx * nstates, (idx + 1) * nstates
        yield modelset[start:end]


def setup(parser):
    parser.add_argument('gsm', help='gsm of which to set the prior')
    parser.add_argument('posts', help='input latent posteriors')
    parser.add_argument('sploop', help='input subspace phone-loop')
    parser.add_argument('out_sploop', help='output subspace phone-loop')


def main(args, logger):
    logger.debug('loading the GSM')
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)

    logger.debug('loading the units posterior')
    with open(args.posts, 'rb') as f:
        latent_posts, nunits, nstates, groupidx = pickle.load(f)

    logger.debug('loading the subspace phoneloop')
    with open(args.sploop, 'rb') as f:
        sploop = pickle.load(f)

    logger.debug('loading the units')
    units_emissions = sploop.modelset.original_modelset.modelsets[groupidx]
    units = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    logger.info(f'initializing the phone loop')
    pdfvecs = gsm.expected_pdfvecs(latent_posts)
    gsm.update_models(units, pdfvecs)

    logger.debug('saving the subspace phoneloop')
    with open(args.out_sploop, 'wb') as f:
        pickle.dump(sploop, f)


if __name__ == "__main__":
    main()

