
'print the most likely path of all the utterances of a dataset'

import argparse
import bisect
from itertools import groupby
import pickle
import sys

import beer


def setup(parser):
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('dataset', help='training data set')
    #parser.add_argument('out', help='output sequence')


def state2phone(path, start_pdf):
    previous_state = path[0]
    start_pdf_list = list(start_pdf.values())
    state2sym = {value: key for key, value in start_pdf.items()}
    phones = [state2sym[previous_state]]
    for state in path[1:]:
        if state != previous_state and state in start_pdf_list:
            phones.append(state2sym[state])
        previous_state = state
    return phones

def main(args, logger):
    logger.debug('load the model')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    logger.debug('load the dataset')
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    count = 0
    for utt in dataset.utterances(random_order=False):
        logger.debug(f'processing utterance: {utt.id}')
        path_ids = [int(unit) for unit in model.decode(utt.features)]
        phones = state2phone(path_ids, model.start_pdf)
        print(utt.id, ' '.join(phones))
        count += 1

    logger.info(f'successfully decoded {count} utterances.')

if __name__ == "__main__":
    main()

