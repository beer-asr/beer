
'create the decoding graph'

import argparse
import pickle
import sys

import beer


def get_first_emitting_state_pdf(graph):
    state_ids = [pdf_id for pdf_id, _ in
                        graph.find_next_pdf_ids(graph.start_state)]
    if len(state_ids) != 1:
        raise ValueError('expected only one start emitting states got: ' \
                         f'{len(state_ids)}')
    return graph.state_from_id(state_ids[0]).pdf_id


def get_last_emitting_state_pdf(graph):
    state_ids = [pdf_id for pdf_id, _ in
                        graph.find_previous_pdf_ids(graph.end_state)]
    if len(state_ids) != 1:
        raise ValueError('expected only one last emitting states got: '\
                         f'{len(state_ids)}')
    return graph.state_from_id(state_ids[0]).pdf_id


def setup(parser):
    parser.add_argument('phoneloop', help='phone loop graph')
    parser.add_argument('hmms', help='phones\' hmms')
    parser.add_argument('out', help='output phone-loop graph')


def main(args, logger):
    logger.debug('loading the phone loop graph...')
    with open(args.phoneloop, 'rb') as f:
        graph = pickle.load(f)

    logger.debug('loading the phones \' hmms...')
    with open(args.hmms, 'rb') as f:
        units, emissions = pickle.load(f)

    logger.debug('build the mapping phone -> state from the symbol table')
    phone2state = {phone: state for state, phone in graph.symbols.items()}

    logger.debug('replace the phone state with the corresponding hmm')
    for phone, hmm in units.items():
        state = phone2state[phone]
        graph.replace_state(state, hmm)

    logger.debug('normalize the graph')
    graph.normalize()

    logger.debug('get the pdf id for the entry/exit state for each phone')
    start_pdf, end_pdf = {}, {}
    for phone, hmm in units.items():
        start_pdf[phone] = get_first_emitting_state_pdf(hmm)
        end_pdf[phone] = get_last_emitting_state_pdf(hmm)

    logger.debug('saving the decoding graph on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump((graph, start_pdf, end_pdf), f)

    logger.info('created decoding graph. ' \
                f'# states: {len(list(graph.states()))} ' \
                f'# arcs: {len(list(graph.arcs()))} ')


if __name__ == "__main__":
    main()

