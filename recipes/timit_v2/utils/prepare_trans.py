

import numpy as np
import argparse
import os

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
        transcription into intergals, and save in npz file')
    parser.add_argument('trans', help='Transcription file')
    parser.add_argument('phonelist', help='phones.txt')
    parser.add_argument('outdir', help='Output directory')
    args = parser.parse_args()

    text = args.trans
    outfile = os.path.join(args.outdir, 'phones.int.npz')
    dict_map = read_phonelist(args.phonelist)
    dict_phones = {}
    with open(text, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            uttid = tokens.pop(0)
            dict_phones[uttid] = np.asarray([dict_map[i] for i in tokens])
    np.savez(outfile, **dict_phones)

if __name__ == '__main__':
    main()
