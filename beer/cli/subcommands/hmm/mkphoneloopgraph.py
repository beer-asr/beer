
'create a phone-loop graph'

import argparse
import pickle
import sys

import beer

START_SYM = '\<s\>'
END_SYM = '\</s\>'
PIVOT_SYM = '#1'


def setup(parser):
    parser.add_argument('-s', '--start-end-group',
                        help='the phone loop start and end by this "group"')
    parser.add_argument('units', help='list of units to build and their ' \
                                      'corresponding group')
    parser.add_argument('out', help='output phone-loop graph')


def main(args, logger):
    logger.debug(f'load the acoustic units name and group')
    with open(args.units, 'r') as f:
        units = []
        for line in f:
            name, group = line.strip().split()
            units.append((name, group))

    logger.debug('create the graph')
    graph = beer.graph.Graph()

    logger.debug('add the start/end/pivot states')
    graph.start_state = graph.add_state()
    graph.end_state = graph.add_state()
    pivot_state = graph.add_state()

    logger.debug('create the states and the unit2state/state2unit mapping')
    unit2state = {
        START_SYM: graph.start_state,
        END_SYM: graph.end_state,
        PIVOT_SYM: pivot_state
    }
    unit2state.update({name: graph.add_state() for name, _ in units})
    state2unit = {state: unit for unit, state in unit2state.items()}

    if args.start_end_group:
        logger.debug(f'using "{args.start_end_group}" group to start/end the phone-loop')
        start_end_units = []
        for name, group in units:
            if group == args.start_end_group:
                start_end_units.append(name)
    else:
        start_end_units = [state2unit[pivot_state]]

    logger.debug('add the arcs to the graph')
    for unit in start_end_units:
        src = graph.start_state
        dest = unit2state[unit]
        graph.add_arc(src, dest)
    for unit in start_end_units:
        src = unit2state[unit]
        dest = graph.end_state
        graph.add_arc(src, dest)
    for unit, _ in units:
        src = pivot_state
        dest = unit2state[unit]
        graph.add_arc(src, dest)
        src = unit2state[unit]
        dest = pivot_state
        graph.add_arc(src, dest)
    graph.symbols = state2unit


    logger.debug('normalize the transitions')
    graph.normalize()

    logger.debug('saving the graph on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump(graph, f)

    logger.info('created phone-loop graph. ' \
                f'# states: {len(list(graph.states()))} ' \
                f'# arcs: {len(list(graph.arcs()))} ' \
                f'start/end group: {args.start_end_group}')


if __name__ == "__main__":
    main()

