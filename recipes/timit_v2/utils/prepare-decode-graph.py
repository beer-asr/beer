
'Create a graph from a txt file.'

import numpy as np
import argparse
import pickle
import logging
import beer

logging.basicConfig(format='%(levelname)s: %(message)s')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_txt', type=str, help='input graph (txt format)')
    parser.add_argument('graph', type=str, help='output graph')
    args = parser.parse_args()

    with open(args.graph_txt, 'r') as fid:
        states = set()
        arcs = set()
        for line in fid:
            tokens = line.split()
            start, end = tokens[0], tokens[1]
            states.add(start)
            states.add(end)
            weight = 1.0
            if len(tokens) == 3:
                weight = float(tokens[2])
            arcs.add((start, end, weight))
    # Create the states
    graph = beer.graph.Graph()
    id2sym, sym2id = {}, {}
    for s_name in sorted(list(states)):
        state_id = graph.add_state()
        id2sym[state_id] = s_name
        sym2id[s_name] = state_id

    # Set the start/end states.
    graph.start_state = sym2id['[s]']
    graph.end_state = sym2id['[/s]']

    # Create the arcs.
    for start, end, weight in arcs:
        graph.add_arc(sym2id[start], sym2id[end], weight)

    graph.symbols = id2sym
    graph.normalize()

    with open(args.graph, 'wb') as fid:
        pickle.dump(graph, fid)


if __name__ == '__main__':
    main()

