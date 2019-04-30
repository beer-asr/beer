
'create a phone-loop model'

import argparse
import pickle
import sys

import torch
import beer


def build_categorical(size):
    mean = torch.ones(size) / size
    return beer.Categorical.create(mean, prior_strength=1)


def build_sb(size):
    return beer.SBCategorical.create(truncation=size, prior_strength=size / 2)

def build_sbhp(size):
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
    parser.add_argument('decode_graph', help='decoding graph')
    parser.add_argument('hmms', help='phones\' hmm')
    parser.add_argument('out', help='phone loop model')


def main(args, logger):
    logger.debug('load the decoding graph...')
    with open(args.decode_graph, 'rb') as f:
        graph, start_pdf, end_pdf = pickle.load(f)

    logger.debug('load the hmms...')
    with open(args.hmms, 'rb') as f:
        hmms, emissions = pickle.load(f)

    logger.debug('compiling the graph...')
    cgraph = graph.compile()

    categorical = priors[args.weights_prior](len(start_pdf))
    logger.debug('create the phone-loop model...')
    ploop = beer.PhoneLoop.create(cgraph, start_pdf, end_pdf, emissions,
                                  categorical)

    logger.debug('saving the model on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(ploop, f)

    logger.info('successfully created a phone-loop model with ' \
                f'{len(start_pdf)} phones')

if __name__ == "__main__":
    main()

