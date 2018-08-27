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
    parser.add_argument('graph', help='new decoding graph')
    parser.add_argument('out', help='ouptut model')
    args = parser.parse_args()

    # Load the HMM.
    with open(args.model, 'rb') as fid:
        model = pickle.load(fid)

    # Load the new decoding graph.
    with open(args.graph, 'rb') as fid:
        graph = pickle.load(fid)

    graph.normalize()
    import pdb; pdb.set_trace()
    model.graph = beer.ConstantParameter(graph.compile())

    # Save the updated hmm.
    with open(args.out, 'wb') as fh:
        pickle.dump(model, fh)


if __name__ == '__main__':
    main()
