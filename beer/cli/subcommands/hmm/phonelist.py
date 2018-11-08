
'print the list of phones from a set of phones\' HMM'

import argparse
import pickle

from natsort import natsorted


def setup(parser):
    parser.add_argument('hmms', help='phones\' hmms')


def main(args, logger):
    logger.debug('loading the HMMs...')
    with open(args.hmms, 'rb') as f:
        units, _ = pickle.load(f)

    for key in natsorted(units.keys(), key=lambda x: x.lower()):
        print(key)


if __name__ == "__main__":
    main()

