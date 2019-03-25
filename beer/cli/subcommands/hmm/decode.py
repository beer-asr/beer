
'print the most likely path of all the utterances of a dataset'

import argparse
import bisect
from itertools import groupby
import pickle
import sys

import numpy as np
import beer


def setup(parser):
    parser.add_argument('-a', '--alis', help='alignment graphs in a "npz" '
                                             'archive')
    parser.add_argument('--per-frame', action='store_true',
                        help='output the per-frame transcription')
    parser.add_argument('-s', '--acoustic-scale', default=1., type=float,
                        help='scaling factor of the acoustic model')
    parser.add_argument('-u', '--utts',
                        help='decode the given utterances ("-") for stdin')
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('dataset', help='training data set')


def state2phone(path, start_pdf, per_frame):
    start_pdf_list = list(start_pdf.values())
    state2sym = {value: key for key, value in start_pdf.items()}
    previous_state = path[0]
    last_sym = state2sym[previous_state]
    phones = [last_sym]
    for state in path[1:]:
        if state != previous_state and state in start_pdf_list:
            last_sym = state2sym[state]
            phones.append(last_sym)
        elif per_frame:
            phones.append(last_sym)
        previous_state = state
    return phones

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
        utt = dataset[uttname]

        aligraph = None
        if alis:
            try:
                aligraph = alis[utt.id][0]
            except KeyError:
                logger.warning(f'no alignment graph for utterance "{utt.id}"')

        logger.debug(f'processing utterance: {utt.id}')
        path_ids = [
            int(unit)
            for unit in model.decode(utt.features, inference_graph=aligraph,
                                     scale=args.acoustic_scale)
        ]
        phones = state2phone(path_ids, model.start_pdf, args.per_frame)
        print(utt.id, ' '.join(phones))
        count += 1

    logger.info(f'successfully decoded {count} utterances.')


if __name__ == "__main__":
    main()

