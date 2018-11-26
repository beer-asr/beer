
'prepare the alignment graphs from a transcription'


import argparse
import pickle
import logging
import os
import sys

import numpy as np
import beer


def create_graph_from_seq(seq, phone_graphs):
    # Create the linear graph corresponding to the sequence.
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


def setup(parser):
    parser.add_argument('trans', help='text transcription or "-" for stdin')
    parser.add_argument('hmms', help='phones\' hmm')
    parser.add_argument('outdir', help='output directory')


def main(args, logger):
    logger.debug('load the phones\' HMM')
    with open(args.hmms, 'rb') as fid:
        hmm_graphs, _ = pickle.load(fid)

    if args.trans == '-':
        ali_txt = sys.stdin
    else:
        with open(args.trans , 'r') as f:
            ali_txt = f.readlines()

    count = 0
    for line in ali_txt:
        tokens = line.strip().split()
        uttid, phones = tokens[0], tokens[1:]
        logger.debug(f'processing utterance: {uttid}')
        graph = create_graph_from_seq(phones, hmm_graphs)

        # Save the alignment graph.
        path = os.path.join(args.outdir, uttid + '.npy')
        graph = np.array([graph])
        np.save(path, graph)

        count += 1

    logger.info(f'create alignment graph for {count} utterance(s)')


if __name__ == '__main__':
    main()

