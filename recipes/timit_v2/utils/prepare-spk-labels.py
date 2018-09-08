'Prepare the speaker labels.'

import argparse
import sys
import numpy as np
import yaml

def read_mapping(infile):
    mapping = {}
    start_pdfs = []
    prev_unit = ''
    with open(infile, 'r') as p:
        for line in p:
            tokens = line.strip().split()
            mapping[tokens[0]] = tokens[1]
            if tokens[1] != prev_unit:
                start_pdfs.append(tokens[0])
                prev_unit = tokens[1]
    return mapping, start_pdfs

def convert_state_to_phone(pdf_ids, start_pdfs, map_pdf_unit):
    '''Convert state ids into phone ids
    Args:
    pdf_ids(list): state ids (str)
    start_pdfs(list): starting state ids (str)
    map_pdf_unit(dict): state id (str) -> unit symbol (str)
    Returns:
    units(list):  unit symbols (str)
    '''
    units = []
    units.append(map_pdf_unit[pdf_ids[0]])
    for i in range(1, len(pdf_ids)):
        if (pdf_ids[i] != pdf_ids[i-1]) and (pdf_ids[i] in start_pdfs):
            units.append(map_pdf_unit[pdf_ids[i]])
    return units

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('utt2spk', help='utterance 2 speaker mapping')
    parser.add_argument('out', help='output archive')
    args = parser.parse_args()


    speakers = set()
    with open(args.utt2spk, 'r') as f:
        for line in f:
            utt, spk = line.strip().split()
            speakers.add(spk)
    speakers = list(speakers)

    data = {}
    with open(args.utt2spk, 'r') as f:
        for line in f:
            utt, spk = line.strip().split()
            data[utt] = np.array([speakers.index(spk)])

    np.savez(args.out, **data)

    for i, spk in enumerate(speakers):
        print(spk, i)


if __name__ == '__main__':
    main()
