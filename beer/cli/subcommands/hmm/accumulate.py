
'Accumulate the ELBO from a list of utterances given from "stdin"'

import argparse
import pickle
import sys

import numpy as np

import beer


def setup(parser):
    parser.add_argument('-a', '--alis', help='alignment graphs in a "npz" '
                                             'archive')
    parser.add_argument('-s', '--acoustic-scale', default=1., type=float,
                        help='scaling factor of the acoutsic model')
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('dataset', help='training data set')
    parser.add_argument('out', help='output accumulated ELBO')


def main(args, logger):
    logger.debug('load the model')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    logger.debug('load the dataset')
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    alis = None
    if args.alis:
        logger.debug('loading alignment graphs')
        alis = np.load(args.alis)

    elbo = beer.evidence_lower_bound(datasize=dataset.size)
    count = 0
    for line in sys.stdin:
        uttid = line.strip().split()[0]
        utt = dataset[uttid]

        aligraph = None
        if alis:
            try:
                aligraph = alis[uttid][0]
            except KeyError:
                logger.warning(f'no alignment graph for utterance "{uttid}"')
        logger.debug(f'processing utterance: {utt.id}')
        elbo += beer.evidence_lower_bound(model, utt.features,
                                          inference_graph=aligraph,
                                          datasize=dataset.size,
                                          scale=args.acoustic_scale)
        logger.debug(f'ELBO: {float(elbo)}')
        count += 1

    logger.debug('saving the accumulated ELBO...')
    with open(args.out, 'wb') as f:
        pickle.dump((elbo, count), f)

    logger.info(f'accumulated ELBO over {count} utterances: {float(elbo) / (count * dataset.size) :.3f}.')

if __name__ == "__main__":
    main()

