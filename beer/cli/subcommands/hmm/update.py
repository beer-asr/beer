
'Update the parameters of the model given the set of ELBO loaded from stdin'

import argparse
import pickle
import sys

import beer


def setup(parser):
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('optimizer', help='parameters\' optimizer')
    parser.add_argument('out_model', help='updated model')
    parser.add_argument('out_optim', help='updated optimizer')


def main(args, logger):
    logger.debug('load the model')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    logger.debug('load the optimizer')
    with open(args.optimizer, 'rb') as f:
        optimizer = pickle.load(f)

    logger.debug('give the model parameters to the optimizer')
    optimizer.groups = model.mean_field_factorization()

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

    # This step is necessary once the elbo object has been stored on
    # disk.
    logger.debug('synchronizing the ELBO and the optimizer')
    elbo.sync(model)

    logger.debug('computing the gradient')
    elbo.backward()

    logger.debug('updating the model')
    optimizer.step()

    logger.debug('preparing the optimizer for the next step')
    optimizer.init_step()

    logger.debug('saving the new model')
    with open(args.out_model, 'wb') as f:
        pickle.dump(model, f)

    logger.debug('saving the updated optimizer')
    with open(args.out_optim, 'wb') as f:
        pickle.dump(optimizer, f)

    logger.info(f'accumulated ELBO={float(elbo)/(nutts * elbo._datasize):.3f}')

if __name__ == "__main__":
    main()

