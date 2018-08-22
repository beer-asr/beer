'Map state alignment to a sequence of unit given a mapping.'

import sys
import argparse
from itertools import groupby


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phone-level', action='store_true',
                        help='Convert the state level alignment to ' \
                             'phone transcription')
    parser.add_argument('mapping', help='pdf to unit mapping')
    args = parser.parse_args()

    with open(args.mapping, 'r') as fh:
        mapping =  {}
        for line in fh:
            tokens = line.strip().split()
            mapping[tokens[0]] = tokens[1]

    for line in sys.stdin:
        tokens = line.strip().split()
        syms = [mapping.get(token, token) for token in tokens]
        if args.phone_level:
            syms = [sym[0] for sym in groupby(syms)]
        print(' '.join(syms), file=sys.stdout)


if __name__ == '__main__':
    main()

