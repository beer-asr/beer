
'print the most likely path of all the utterances of a dataset'

import argparse
import os
import pickle
import sys

import numpy as np
import beer


EPS = 1e-5


def setup(parser):
    parser.add_argument('-S', '--state', action='store_true',
                        help='state level posteriors')
    parser.add_argument('-l', '--log', action='store_true',
                        help='log domain')
    parser.add_argument('-s', '--acoustic-scale', default=1., type=float,
                        help='scaling factor of the acoustic model')
    parser.add_argument('-u', '--utts',
                        help='decode the given utterances ("-") for stdin')
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('dataset', help='training data set')
    parser.add_argument('outdir', help='output directory')


def state2phone(posts, start_pdf, end_pdf):
    retval = np.zeros((len(posts), len(start_pdf)))
    for i, unit in enumerate(start_pdf):
        start, end = start_pdf[unit], end_pdf[unit]
        retval[:, i] = posts[:, start:end].sum(axis=-1)
    return retval


def main(args, logger):
    logger.debug('load the model')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    logger.debug('load the dataset')
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    if args.utts:
        if args.utts == '-':
            utts = [line.strip().split()[0] for line in sys.stdin.readlines()]
        else:
            with open(args.utts, 'r') as f:
                utts = [line.strip().split()[0] for line in f.readlines()]
    else:
        utts = list([utt.id for utt in dataset.utterances(random_order=False)])

    count = 0
    for uttname in utts:
        try:
            utt = dataset[uttname]
        except KeyError as err:
            logger.warning(f'no data for utterance {uttname}')
            continue
        logger.debug(f'processing utterance: {utt.id}')
        posts = model.posteriors(utt.features, scale=args.acoustic_scale)
        posts = posts.detach().numpy()
        if not args.state:
            posts = state2phone(posts, model.start_pdf, model.end_pdf)
        if args.log:
            posts = np.log(EPS + posts)
        path = os.path.join(args.outdir, f'{uttname}.npy')
        np.save(path, posts)
        count += 1

    logger.info(f'successfully computed the posteriors for {count} utterances.')


if __name__ == "__main__":
    main()

