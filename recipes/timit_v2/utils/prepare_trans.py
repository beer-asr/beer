

import numpy as np
import argparse
import os
import sys

def read_phonelist(infile):
    '''Read phones.txt file and store in a dictionary
    Args:
        infile(str): file with phones
    Returns:
        dict_phone: phoneme(str) -> phone index(int)
    '''
    dict_map = {}
    with open(infile, 'r') as p:
        for line in p:
            tokens = line.strip().split()
            dict_map[tokens[0]] = int(tokens[1])
    return dict_map

def main():
    parser = argparse.ArgumentParser(description='Convert text \
        transcriptions into integer sequences, and save in npz file')
    parser.add_argument('phonelist', help='phones.txt')
    parser.add_argument('out_npz', help='output numpy archive')
    args = parser.parse_args()

    dict_map = read_phonelist(args.phonelist)
    dict_phones = {}
    for line in sys.stdin:
        tokens = line.strip().split()
        uttid = tokens.pop(0)
        dict_phones[uttid] = np.asarray([dict_map[i] for i in tokens])
    np.savez(args.out_npz, **dict_phones)

if __name__ == '__main__':
    main()
