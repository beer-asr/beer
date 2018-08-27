'Create a the HMM from a graph and the corresponding emssions.'

import numpy as np
import argparse
import beer
import pickle
import torch
import yaml
import logging

logging.basicConfig(format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('decoding_graph', help='decoding graph')
    parser.add_argument('hmm_graphs', help='hmm graph for each unit')
    parser.add_argument('emissions', help='emissions')
    parser.add_argument('hmm', help='output hmm')
    args = parser.parse_args()

    with open(args.decoding_graph, 'rb') as fid:
        decoding_graph = pickle.load(fid)
    with open(args.hmm_graphs, 'rb') as fid:
        hmm_graphs = pickle.load(fid)
    with open(args.emissions, 'rb') as fid:
        emissions = pickle.load(fid)

    id2sym = decoding_graph.symbols
    sym2id = {val: key for key, val in id2sym.items()}
    for unit, unit_graph in hmm_graphs.items():
        decoding_graph.replace_state(sym2id[unit], unit_graph)
    cgraph = decoding_graph.compile()
    hmm = beer.HMM.create(cgraph, emissions)

    with open(args.hmm, 'wb') as fid:
        pickle.dump(hmm, fid)

if __name__ == '__main__':
    main()
