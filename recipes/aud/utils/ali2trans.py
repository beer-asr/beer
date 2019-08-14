'convert time alignments to a transcription.'

import argparse
from itertools import groupby
import sys


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ali', help='time alignment')
    args = parser.parse_args()

    with open(args.ali, 'r') as f:
        for line in f:
            seq = [x[0] for x in groupby(line.strip().split())]
            print(' '.join(seq), file=sys.stdout)


if __name__ == '__main__':
    run()

