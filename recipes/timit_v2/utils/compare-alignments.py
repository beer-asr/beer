'Estimate the confusion matrix between two set of forced alignments.'

import argparse
from collections import defaultdict
from functools import partial
import logging
import pickle

import numpy as np

logging.basicConfig(format='%(levelname)s: %(message)s')


def load_pdf_mapping(fname):
    pdf_mapping = {}
    with open(fname, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            pdf_mapping[int(tokens[0])] = tokens[1]
    return pdf_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', action='store_true',
                        help='show debug messages')
    parser.add_argument('--epsilon', type=float, default=1e-12,
                        help='small constant to add to the confusion ' \
                             'matrix to avoid overflow')
    parser.add_argument('--groups', help='list of ref symbol group to be '\
                                         'merged together')
    parser.add_argument('ref_alis', help='reference alignments')
    parser.add_argument('ref_pdf_mapping', help='pdf mapping for the ' \
                                                'reference alignments')
    parser.add_argument('hyp_alis', help='alignments')
    parser.add_argument('hyp_pdf_mapping', help='pdf mapping for the ' \
                                                'alignments')
    parser.add_argument('out', help='normalized confusion matrix (and ' \
                                    'marginal distributions)')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ref_alis = np.load(args.ref_alis)
    ref_pdf_mapping = load_pdf_mapping(args.ref_pdf_mapping)
    hyp_alis = np.load(args.hyp_alis)
    hyp_pdf_mapping = load_pdf_mapping(args.hyp_pdf_mapping)

    joint_counts = defaultdict(partial(defaultdict, partial(float, args.epsilon)))
    ref_counts = defaultdict(partial(float, args.epsilon))
    hyp_counts = defaultdict(partial(float, args.epsilon))
    for file in ref_alis.files:
        ref_ali = ref_alis[file]
        try:
            hyp_ali = hyp_alis[file]
        except KeyError:
            logging.warning(f'{file} is in reference alignments but not ' \
                            'not the hypothesis alignments')

        logging.debug(f'processing alignments for utterance {file}')
        for ref_state, hyp_state in zip(ref_ali, hyp_ali):
            ref_sym, hyp_sym = ref_pdf_mapping[ref_state], \
                               hyp_pdf_mapping[hyp_state]
            ref_counts[ref_sym] += 1
            hyp_counts[hyp_sym] += 1
            joint_counts[ref_sym][hyp_sym] += 1

    # Remap the confusion matrix.
    if args.groups:
        logging.debug(f'grouping the confusion matrix using: {args.groups}')
        groups = {}
        with open(args.groups, 'r') as f:
            for line in f:
                name, syms = line.strip().split(':')
                groups[name] = tuple(syms.strip().split())
        mapped_joint_counts = defaultdict(partial(defaultdict, partial(float, args.epsilon)))
        mapped_ref_counts = defaultdict(partial(float, args.epsilon))
        for name in groups:
            for ref_sym in groups[name]:
                for hyp_sym in joint_counts[ref_sym]:
                    mapped_joint_counts[name][hyp_sym] += \
                        joint_counts[ref_sym][hyp_sym]
                mapped_ref_counts[name] += ref_counts[ref_sym]
        joint_counts = mapped_joint_counts
        ref_counts = mapped_ref_counts

    #import pdb; pdb.set_trace()

    with open(args.out, 'wb') as f:
        pickle.dump((joint_counts, ref_counts, hyp_counts), f)


if __name__ == "__main__":
    main()
