'''Given a set of features and the corresponding alignments,
extract the features frame corresponding to the center of each
token in the alignments.'''

import argparse
import glob
import os
import sys

import numpy as np


def get_cf(ali, pdf_mapping):
    start = 0
    previous_lab = pdf_mapping[ali[0]]
    cfs= []
    for i, pdf_id in enumerate(ali):
        label = pdf_mapping[pdf_id]
        if label != previous_lab:
            cf = int(start + .5 * (i - start))
            cfs.append((previous_lab, cf))
            start = i + 1
            previous_lab = label
    cf = int(start + .5 * (i - start))
    cfs.append((previous_lab, cf))
    return cfs

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phone_mapping', help='phone to id mapping')
    parser.add_argument('ali', help='alignments')
    parser.add_argument('pdf_mapping', help='pdf_id mapping')
    parser.add_argument('fea', help='features')
    parser.add_argument('out', help='output numpy archive (npz)')
    args = parser.parse_args()

    ali = np.load(args.ali)
    fea = np.load(args.fea)

    phone_mapping = {}
    with open(args.phone_mapping, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            phone_mapping[tokens[0]] = int(tokens[1])

    pdf_mapping = {}
    with open(args.pdf_mapping, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            pdf_mapping[int(tokens[0])] = tokens[1]

    new_fea = []
    labels = []
    for line in sys.stdin:
        uttid = line.strip()
        utt_fea, utt_ali = fea[uttid], ali[uttid]
        cfs = get_cf(utt_ali, pdf_mapping)
        for label, cf in cfs:
            new_fea.append(utt_fea[cf])
            labels.append(phone_mapping[label])

    new_fea = np.array(new_fea)
    labels = np.array(labels, dtype=int)

    np.savez(args.out, features=new_fea, labels=labels)

if __name__ == '__main__':
    run()

