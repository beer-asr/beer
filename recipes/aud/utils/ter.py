'Token Error Rate'

import argparse
from collections import defaultdict
from itertools import groupby
import random
import sys
import yaml

import numpy as np

def load_transcript(path):
    with open(path, 'r') as f:
        trans = {}
        for line in f:
            tokens = line.strip().split()
            trans[tokens[0]] = tokens[1:]
    return trans


def wer(r, h):
    '''
    Calculation of WER with Levenshtein distance.

    Code from: https://martin-thoma.com/word-error-rate-calculation/

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    '''
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=int)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def load_mapping(path):
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                p1, p2 = line.strip().split()
                mapping[p1] = p2
            except ValueError:
                pass
    return mapping


def map_trans(trans, mapfile):
    new_trans = {}
    for utt, utt_trans in trans.items():
        tmp = []
        for token in utt_trans:
            try:
                new_token = mapfile[token]
                tmp.append(new_token)
            except KeyError:
                pass
        new_trans[utt] = tmp
    return new_trans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', action='store_true',
                        help='show random examples on stderr')
    parser.add_argument('--no-repeat', action='store_false',
                        help='merge consecutive duplicate into one token')
    parser.add_argument('--mapping', help='mapping')
    parser.add_argument('ref', help='reference transcription')
    parser.add_argument('hyp', help='proposed transcription')
    args = parser.parse_args()

    ref = load_transcript(args.ref)
    hyp = load_transcript(args.hyp)
    if args.mapping:
        mapping = load_mapping(args.mapping)
        ref = map_trans(ref, mapping)
        hyp = map_trans(hyp, mapping)
    if not args.no_repeat:
        hyp = {utt: [x[0] for x in groupby(trans)]
               for utt, trans in hyp.items()}
        ref = {utt: [x[0] for x in groupby(trans)]
               for utt, trans in ref.items()}

    if args.examples:
        print('Random example:', file=sys.stderr)
        for utt in random.choices(list(ref.keys()), k=5):
            print('(ref)', utt, ' '.join(ref[utt]), file=sys.stderr)
            print('(hyp)', utt, ' '.join(hyp[utt]), file=sys.stderr)
            print('---------------')

    acc_wer = 0
    nwords = 0
    for utt in ref:
        try:
            ref_t, hyp_t = ref[utt], hyp[utt]
            acc_wer += wer(ref_t, hyp_t)
            nwords += len(ref_t)
        except KeyError:
            pass
    print('TER (%)')
    print(f'{100 * acc_wer / nwords:.2f}')


if __name__ == '__main__':
    main()

