
'Add a prefix to all token in a transcription.'

import argparse
import os
import pickle
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exclude', default='',
                        help='comma separated list of token to leave unchanged')
    parser.add_argument('trans', help='transcription')
    parser.add_argument('prefix', help='prefix to append')
    args = parser.parse_args()

    to_exclude = args.exclude.split(',')
    with open(args.trans, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            uttid, trans = tokens[0], tokens[1:]
            new_trans = []
            for token in trans:
                if token not in to_exclude:
                    new_trans.append(f'{args.prefix}{token}')
                else:
                    new_trans.append(token)
            print(uttid, ' '.join(new_trans))


if __name__ == "__main__":
    main()

