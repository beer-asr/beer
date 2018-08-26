
'Create the phone loop decoding graph in text format.'

import numpy as np
import argparse
import pickle
import logging
import beer

logging.basicConfig(format='%(levelname)s: %(message)s')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unigram-lm', help='unigram language model')
    parser.add_argument('--use-silence', action='store_true',
                        help='put the silence at the beginning/end ' \
                             'of the graph. Assume silence is the ' \
                             'is the first element of the phone list')
    parser.add_argument('phones', type=str, help='list of phones')

    args = parser.parse_args()

    phones = []
    with open(args.phones, 'r') as fid:
        for line in fid:
            tokens = line.split()
            phones.append(tokens[0])

    weights = np.ones(len(phones))
    if args.unigram_lm:
        with open(args.unigram_lm, 'rb') as fh:
            lm = pickle.load(fh)
            weights = lm.weights.expected_value().numpy()

    joint_state = '_1'

    if args.use_silence:
        silence = phones[0]
        print('[s]', silence)
        print(silence, '[/s]')
        print(silence, joint_state)
        print(joint_state, silence, weights[0])
        start_idx = 1
    else:
        print('[s]', joint_state)
        print(joint_state, '[/s]')
        start_idx = 0

    for i, phone in enumerate(phones[start_idx:]):
        print(joint_state, phone, weights[start_idx + i])
        print(phone, joint_state)

if __name__ == '__main__':
    main()
