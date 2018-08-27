
'Print the mapping pdf <-> unit from a set of hmms.'

import numpy as np
import argparse
import pickle
import logging
import beer

logging.basicConfig(format='%(levelname)s: %(message)s')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graphs', help='set of unit hmm')
    args = parser.parse_args()

    with open(args.graphs, 'rb') as fh:
        graphs = pickle.load(fh)

    for unit, graph in graphs.items():
        for state_id in graph.states():
            pdf_id = graph._states[state_id].pdf_id
            if pdf_id is not None:
                print(pdf_id, unit)

if __name__ == '__main__':
    main()
