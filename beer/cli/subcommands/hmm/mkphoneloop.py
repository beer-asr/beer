
'create a phone-loop model'

import argparse
import pickle
import sys

import torch
import beer


def build_categorical(size):
    mean = torch.ones(size) / size
    return beer.Categorical.create(mean, prior_strength=1)

def build_categorical_2g(size):
    mean = torch.ones(size, size) / size
    return beer.CategoricalSet.create(mean, prior_strength=1)

def build_sb(size):
    return beer.SBCategorical.create(truncation=size, prior_strength=size / 2)

def build_hsb(size):
    root_sb = beer.SBCategorical.create(truncation=size, 
                                        prior_strength=size / 2)
    return beer.SBCategoricalSet.create(size, root_sb, 
                                        prior_strength=size / 2)
    

def build_sbhp(size):
    return beer.SBCategoricalHyperPrior.create(truncation=size,
                                               prior_strength=size / 2,
                                               hyper_prior_strength=1.)


bigram_prior = ['hierarchical_dirichlet_process', 'dirichlet2']

priors = {
    'dirichlet': build_categorical,
    'dirichlet2': build_categorical_2g,
    'dirichlet_process': build_sb,
    'gamma_dirichlet_process': build_sbhp,
    'hierarchical_dirichlet_process': build_hsb,
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

    model_cls = beer.BigramPhoneLoop if args.weights_prior in bigram_prior else beer.PhoneLoop
    ploop = model_cls.create(cgraph, start_pdf, end_pdf, emissions, categorical)

    logger.debug('saving the model on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(ploop, f)

    logger.info('successfully created a phone-loop model with ' \
                f'{len(start_pdf)} phones')


if __name__ == "__main__":
    main()

