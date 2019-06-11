'convert time alignment to kaldi format'

import argparse
import os
import sys


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('alidir', help='directory containing the alignment')
    args = parser.parse_args()


    for line in sys.stdin:
        uttid = line.strip()
        with open(os.path.join(args.alidir, uttid + '.algn')) as f:
            alis = []
            for line in f:
                tokens = line.strip().split()
                start, end, unit = int(tokens[0]), int(tokens[1]), 'au'+tokens[2]
                alis += [unit] * (end - start + 1)
            print(uttid, ' '.join(alis))


if __name__ == '__main__':
    run()
