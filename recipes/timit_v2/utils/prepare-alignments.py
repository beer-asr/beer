'Create the alignment graph for the HMM training.'


import argparse
import pickle
import logging
import os
import sys

import numpy as np
import beer

logging.basicConfig(format='%(levelname)s: %(message)s')


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true',
                        help='show debug messages')
    parser.add_argument('hmm_graphs', help='hmm graph for each unit')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    with open(args.hmm_graphs, 'rb') as fid:
        hmm_graphs = pickle.load(fid)

    graph = beer.graph.Graph()
    for line in sys.stdin:
        tokens = line.strip().split()
        uttid, phones = tokens[0], tokens[1:]
        logging.debug('Create alignment graph for utterance: {}'.format(uttid))
        graph = create_graph_from_seq(phones, hmm_graphs)

        path = os.path.join(args.outdir, uttid + '.npy')
        graph = np.array([graph])
        np.save(path, graph)

    with open('test.pkl', 'wb') as fid:
        pickle.dump(graph, fid)



if __name__ == '__main__':
    main()
