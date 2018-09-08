'Set the decoding graph of a HMM.'

import numpy as np
import argparse
import beer
import pickle
import torch
import logging

logging.basicConfig(format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='input model')
    parser.add_argument('decoding_graph', help='new decoding graph')
    parser.add_argument('hmm_graphs', help='hmm graph for each unit')
    parser.add_argument('out', help='ouptut model')
    args = parser.parse_args()

    # Load the HMM.
    with open(args.model, 'rb') as fid:
        model = pickle.load(fid)

    with open(args.decoding_graph, 'rb') as fid:
        decoding_graph = pickle.load(fid)
    with open(args.hmm_graphs, 'rb') as fid:
        hmm_graphs = pickle.load(fid)

    id2sym = decoding_graph.symbols
    sym2id = {val: key for key, val in id2sym.items()}
    for unit, unit_graph in hmm_graphs.items():
        decoding_graph.replace_state(sym2id[unit], unit_graph)
    model.latent_model.graph = beer.ConstantParameter(decoding_graph.compile())

    # Save the updated hmm.
    with open(args.out, 'wb') as fh:
        pickle.dump(model, fh)


if __name__ == '__main__':
    main()
