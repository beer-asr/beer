
'train a HMM based model'

import argparse
import pickle
import sys

import beer


def setup(parser):
    parser.add_argument('-b', '--batch-size', type=int, default=-1,
                        help='batch size in number of utterance ' \
                             '(-1 means all the utterances as one batch)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('-l', '--lrate', type=float, default=1.,
                        help='learning rate')
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('dataset', help='training data set')
    parser.add_argument('out', help='phone loop model')


def main(args, logger):
    logger.debug('load the model')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    logger.debug('load the dataset')
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    logger.debug('create the optimizer')
    optim = beer.BayesianModelOptimizer(model.mean_field_factorization(),
                                        lrate=args.lrate)

    for epoch in range(1, args.epochs + 1):
        elbo = beer.evidence_lower_bound(datasize=dataset.size)
        optim.init_step()
        for i, utt in enumerate(dataset.utterances(), start=1):
            logger.debug(f'processing utterance: {utt.id}')
            elbo += beer.evidence_lower_bound(model, utt.features,
                                              datasize=dataset.size)

            # Update the model after N utterances.
            if i % args.batch_size == 0:
                elbo.backward()
                optim.step()
                logger.info(f'epoch: {epoch: <5}  ' \
                            f'batch: {i // args.batch_size }/{int(len(dataset) / args.batch_size): <10} ' \
                            f'ELBO: {float(elbo) / (args.batch_size * dataset.size):<10.3f}')
                elbo = beer.evidence_lower_bound(datasize=dataset.size)
                optim.init_step()
            break
        break

    logger.debug('save the model on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f'finished training after {args.epochs} epochs.' \
                f'KL(q || p) = {float(model.kl_div_posterior_prior()): .3f}')

if __name__ == "__main__":
    main()

