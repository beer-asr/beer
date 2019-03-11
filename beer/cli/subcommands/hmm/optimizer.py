
'create an optimizer with the model parameters'

import argparse
import pickle
import sys

import beer


def setup(parser):
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('optimizer', help='output parameters\' optimizer')


def main(args, logger):
    logger.debug('load the model')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    optim = beer.VariationalBayesOptimizer(model.mean_field_factorization())
    optim.init_step()

    logger.debug('saving the optimizer')
    with open(args.optimizer, 'wb') as f:
        pickle.dump(optim, f)

if __name__ == "__main__":
    main()

