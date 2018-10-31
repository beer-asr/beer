
'create a phone-loop model'

import argparse
import pickle
import sys

import beer


def setup(parser):
    parser.add_argument('phone_loop_graph', help='phone-loop decoding graph')
    parser.add_argument('hmms', help='phones\' hmm')
    parser.add_argument('out', help='phone loop model')


def main(args, logger):
    logger.debug('load the decoding graph...')
    with open(args.phone_loop_graph, 'rb') as f:
        graph, start_pdf, end_pdf = pickle.load(f)

    logger.debug('load the hmms...')
    with open(args.hmms, 'rb') as f:
        hmms, emissions = pickle.load(f)

    logger.debug('compiling the graph...')
    cgraph = graph.compile()

    logger.debug('create the phone-loop model...')
    ploop = beer.PhoneLoop.create(cgraph, start_pdf, end_pdf, emissions)

    logger.debug('saving the model on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(ploop, f)


    logger.info('successfully created a phone-loop model with ' \
                f'{len(start_pdf)} phones')

if __name__ == "__main__":
    main()

