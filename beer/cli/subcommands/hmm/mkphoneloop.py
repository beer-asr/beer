
'create a phone-loop graph'

import argparse
import pickle
import sys

import beer


START_SYM = '\<s\>'
END_SYM = '\</s\>'
PIVOT_SYM = '#1'


def setup(parser):
    parser.add_argument('-s', '--sil-prefix',
                        help='prefix for the silence phones')
    parser.add_argument('phone_list', help='list of phones files or "-" '\
                        'for stdin')
    parser.add_argument('out', help='output phone-loop graph')


def main(args, logger):
    logger.debug('loading the phone list...')
    if args.phone_list == '-':
        infile = sys.stdin
    else:
        with open(args.phone_list, 'r') as f:
            infile = f.readlines()
    phones = [line.strip() for line in infile]

    logger.debug('create the graph')
    graph = beer.graph.Graph()

    logger.debug('add the start/end/pivot states')
    graph.start_state = graph.add_state()
    graph.end_state = graph.add_state()
    pivot_state = graph.add_state()

    logger.debug('create the states and the phone2state/state2phone mapping')
    phone2state = {
        START_SYM: graph.start_state,
        END_SYM: graph.end_state,
        PIVOT_SYM: pivot_state
    }
    phone2state.update({phone: graph.add_state() for phone in phones})
    state2phone = {state:phone for phone, state in phone2state.items()}


    if args.sil_prefix:
        logger.debug(f'using "{args.sil_prefix}*" phones as start/end states')
        starting_phones = []
        ending_phones = []
        for phone in phones:
            if phone.startswith(args.sil_prefix):
                starting_phones.append(phone)
                ending_phones.append(phone)
    else:
        starting_phones = [state2phone[pivot_state]]
        ending_phones = [state2phone[pivot_state]]

    logger.debug('add the arcs to the graph')
    for phone in starting_phones:
        src = graph.start_state
        dest = phone2state[phone]
        graph.add_arc(src, dest)
    for phone in ending_phones:
        src = phone2state[phone]
        dest = graph.end_state
        graph.add_arc(src, dest)
    for phone in phones:
        src = pivot_state
        dest = phone2state[phone]
        graph.add_arc(src, dest)
        src = phone2state[phone]
        dest = pivot_state
        graph.add_arc(src, dest)
    graph.symbols = state2phone


    logger.debug('normalize the transitions')
    graph.normalize()

    logger.debug('saving the graph on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(graph, f)

    logger.info('created phone-loop graph. ' \
                f'# states: {len(list(graph.states()))} ' \
                f'# arcs: {len(list(graph.arcs()))} ' \
                f'silence phone prefix: {args.sil_prefix}')


if __name__ == "__main__":
    main()

