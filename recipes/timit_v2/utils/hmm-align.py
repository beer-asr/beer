
'Estimate the state alignments.'

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch

import beer


log_format = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ali-graphs', help='aligment graph for each ' \
                                             'utterance')
    parser.add_argument('hmm', help='hmm model to train')
    parser.add_argument('feats', help='Feature file')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    # Load the data for the training.
    feats = np.load(args.feats)

    ali_graphs = None
    if args.ali_graphs:
        ali_graphs = np.load(args.ali_graphs)

    with open(args.hmm, 'rb') as fh:
        model = pickle.load(fh)

    for line in sys.stdin:
        uttid = line.strip()
        ft = torch.from_numpy(feats[uttid]).float()
        graph = None
        if ali_graphs is not None:
            graph = ali_graphs[uttid][0]
        ali = model.decode(ft, inference_graph=graph)
        path = os.path.join(args.outdir, uttid + '.npy')
        np.save(path, ali.numpy())


if __name__ == "__main__":
    main()
