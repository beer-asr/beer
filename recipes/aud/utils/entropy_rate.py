
'Empirical entropy rate of a transcription.'


import argparse
from collections import defaultdict
from itertools import groupby
import numpy as np

def load_transcript(path):
    with open(path, 'r') as f:
        trans = {}
        for line in f:
            tokens = line.strip().split()
            trans[tokens[0]] = [x[0] for x in groupby(tokens[1:])]
    return trans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trans', help='transcription')
    args = parser.parse_args()

    trans = load_transcript(args.trans)

    counts = defaultdict(int)
    for utt, utt_trans in trans.items():
        for token in utt_trans:
            counts[token] += 1

    prob = np.array([c for _, c in counts.items()], dtype=float)
    prob /= prob.sum()
    entropy = - prob @ np.log2(prob)

    print('entropy rate (bits), perplexity')
    print(f'{entropy:.2f}, {2**entropy:.2f}')


if __name__ == "__main__":
    main()

