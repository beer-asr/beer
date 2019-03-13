
'Update the parameters of the model given the set of ELBO loaded from stdin'

import argparse
import os
import pickle
import sys

import torch
import beer


def setup(parser):
    parser.add_argument('-l', '--learning-rate', default=1., type=float,
                        help='learning rate')
    parser.add_argument('-o', '--optim-state', help='optimizer state')
    parser.add_argument('model', help='model to update')
    parser.add_argument('out_model', help='updated model')


def main(args, logger):
    logger.debug('load the model')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    logger.debug('building the optimizer')
    optim = beer.VBConjugateOptimizer(
        model.conjugate_bayesian_parameters(keepgroups=True),
        lrate=args.learning_rate
    )

    if args.optim_state and os.path.isfile(args.optim_state):
        logger.debug(f'loading optimizer state from: {args.optim_state}')
        state = torch.load(args.optim_state)
        optim.load_state_dict(state)

    optim.init_step()

    elbo = None
    nutts = 0
    count = 0
    for line in sys.stdin:
        path = line.strip()
        logger.debug(f'loading ELBO stored in {path}')
        with open(path, 'rb') as f:
            elbo_batch, nutts_batch = pickle.load(f)
        if elbo is None:
            elbo = elbo_batch
            nutts = nutts_batch
        else:
            elbo += elbo_batch
            nutts += nutts_batch
        count += 1

    logger.debug('synchronizing the ELBO and the model')
    elbo.sync(model)

    logger.debug('computing the gradient')
    elbo.backward()

    logger.debug('updating the model')
    optim.step()

    logger.debug('saving the new model')
    with open(args.out_model, 'wb') as f:
        pickle.dump(model, f)

    if args.optim_state:
        logger.debug(f'saving the optimizer state to: {args.optim_state}')
        torch.save(optim.state_dict(), args.optim_state)

    logger.info(f'accumulated ELBO={float(elbo)/(nutts * elbo._datasize):.3f}')

if __name__ == "__main__":
    main()

