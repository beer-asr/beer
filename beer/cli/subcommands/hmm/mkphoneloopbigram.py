
'create a bigram phone-loop model'

import argparse
import pickle
import sys

import torch
import beer


def build_categorical(size, unigram):
    mean = torch.ones(size, size) / size
    return beer.CategoricalSet.create(mean, prior_strength=1)


def build_sb(size, unigram):
    return beer.SBCategoricalSet.create(truncation=size, prior_strength=size / 2)

def build_sbhp(size, unigram):
    return beer.SBCategoricalHyperPrior.create(truncation=size,
                                               prior_strength=size / 2,
                                               hyper_prior_strength=1.)


priors = {
    'dirichlet': build_categorical,
    'dirichlet_process': build_sb,
    'gamma_dirichlet_process': build_sbhp
}


def setup(parser):
    parser.add_argument('--weights-prior', default='gamma_dirichlet_process',
                        choices=[key for key in priors],
                        help='type of prior over the phone weights')
    parser.add_argument('phoneloop', help='unigram phone loop')
    parser.add_argument('out', help='bigram phone loop (out)')


def main(args, logger):
    logger.debug('load the unigram phone-loop...')
    with open(args.phoneloop, 'rb') as f:
        ploop = pickle.load(f)

    categoricalset = priors[args.weights_prior](len(ploop.start_pdf), 
                                                ploop.categorical)
    logger.debug('create the bigram phone-loop model...')
    ploop2 = beer.BigramPhoneLoop.create(ploop.graph, ploop.start_pdf, 
                                         ploop.end_pdf, ploop.modelset,
                                         categoricalset)

    logger.debug('saving the model on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(ploop2, f)

    logger.info('successfully created a bigram phone-loop model with ' \
                f'{len(ploop.start_pdf)} phones')

if __name__ == "__main__":
    main()
