
'Convert the alignment to the best path.'

import argparse
import logging
import os
import pickle
import sys
from itertools import groupby

import numpy as np
import torch

import beer


log_format = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ali', help='forced alignments')
    parser.add_argument('pdf_mapping', help='mapping of the input data ' \
                                             '(pdf id) to the vocabulary space')
    parser.add_argument('units', help='units list (i.e. phones)')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    # Load the alignments.
    ali = np.load(args.ali)

    # Load the pdf to unit mapping.
    pdf_mapping= {}
    with open(args.pdf_mapping, 'r') as fh:
        for line in fh:
            tokens = line.strip().split()
            pdf_mapping[int(tokens[0])] = tokens[1]

    units_id = {}
    with open(args.units, 'r') as fh:
        for line in fh:
            tokens = line.strip().split()
            units_id[tokens[0]] = int(tokens[1])

    for utt in ali:
        best_state_path = [units_id[pdf_mapping[state]]
                           for state in ali[utt]]
        best_path = np.asarray([s for s, _ in groupby(best_state_path)])
        path = os.path.join(args.outdir, utt + str('.npy'))
        np.save(path, best_path)

if __name__ == "__main__":
    main()
