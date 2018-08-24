
'Create the phone loop decoding graph in text format.'

import numpy as np
import argparse
import pickle
import logging
import beer

logging.basicConfig(format='%(levelname)s: %(message)s')



def main():
    parser = argparse.ArgumentParser()
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
    phones = phones

    joint_state = '_1'

    if args.use_silence:
        silence = phones[0]
        del phones[0]
        print('[s]', silence)
        print(silence, '[/s]')
        print(silence, joint_state)
        print(joint_state, silence)
    else:
        print('[s]', joint_state)
        print(joint_state, '[/s]')

    for phone in phones:
        print(joint_state, phone)
        print(phone, joint_state)

if __name__ == '__main__':
    main()
