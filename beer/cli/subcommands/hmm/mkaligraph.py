'create the alignment graph for the HMM training from a transcription (stdin)'

import argparse
import pickle
import os
import sys

import numpy as np
import beer


def setup(parser):
    parser.add_argument('hmms', help='hmm graph for each unit')
    parser.add_argument('outdir', help='output directory')


# Create the linear graph corresponding to a sequence of phones.
def create_graph_from_seq(seq, phone_graphs):
    graph = beer.graph.Graph()
    graph.start_state = graph.add_state()
    last_state = graph.start_state
    id2sym = {}
    phone_states = []
    for i, phone in enumerate(seq):
        state = graph.add_state()
        phone_states.append(state)
        graph.add_arc(last_state, state)
        last_state = state
        id2sym[i] = phone
    state = graph.add_state()
    graph.add_arc(last_state, state)
    graph.end_state = state

    # Replace the phone states with the corresponding HMMs.
    for i, phone in enumerate(seq):
        graph.replace_state(phone_states[i], phone_graphs[phone])
    graph.normalize()

    return graph.compile()


def main(args, logger):

    logger.debug('loading the hmms')
    with open(args.hmms, 'rb') as fid:
        hmm_graphs, _ = pickle.load(fid)

    nutts = 0
    for line in sys.stdin:
        tokens = line.strip().split()
        uttid, phones = tokens[0], tokens[1:]

        if len(phones) == 0:
            logger.error(f'utterance {uttid} has no transcription')
            continue

        logger.debug(f'create alignment graph for utterance: {uttid}')
        graph = create_graph_from_seq(phones, hmm_graphs)

        path = os.path.join(args.outdir, uttid + '.npy')
        logger.debug(f'storing the alignment graph in numpy format ({path})')
        graph = np.array([graph])
        np.save(path, graph)

        nutts += 1

    logger.info(f'created alignment graphs for {nutts} utterances')


if __name__ == '__main__':
    main()

