'Map state alignment to a sequence of unit given a pdf-to-unit mapping'

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
    parser.add_argument('--phone-level', action='store_true',
                        help='Covert the state level alignments to \
                        phone level transcriptions')
    parser.add_argument('map_pdf_to_phone', help='File: pdf_mapping.txt')
    args = parser.parse_args()
    
    phone_level = args.phone_level
    map_pdf_to_phone, start_pdfs = read_mapping(args.map_pdf_to_phone)

    for line in sys.stdin:
        tokens = line.strip().split()
        if len(tokens) < 2:
            sys.exit('Pdf sequence is empty !')
        uttid = tokens.pop(0)
        phones = [map_pdf_to_phone[k] for k in tokens]
        if phone_level:
            phones = convert_state_to_phone(tokens, start_pdfs, map_pdf_to_phone)
        print(uttid, ' '.join(phones), file=sys.stdout)

if __name__ == '__main__':
    main()
