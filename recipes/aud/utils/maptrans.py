'Transform a transcription given a mapping file.'

import argparse
from collections import defaultdict
import sys
import yaml


def load_transcript(path):
    with open(path, 'r') as f:
        trans = {}
        for line in f:
            tokens = line.strip().split()
            trans[tokens[0]] = tokens[1:]
    return trans


def load_mapping(path):
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                p1, p2 = line.strip().split()
                mapping[p1] = p2
            except ValueError:
                pass
    return mapping


def map_trans(trans, mapfile, unk):
    new_trans = {}
    for utt, utt_trans in trans.items():
        tmp = []
        for token in utt_trans:
            try:
                new_token = mapfile[token]
                tmp.append(new_token)
            except KeyError as err:
                if unk != '':
                    tmp.append(unk)
                else:
                    raise err
        new_trans[utt] = tmp
    return new_trans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unk', default='', help='replace unkonwn mapping')
    parser.add_argument('mapping', help='mapping')
    parser.add_argument('trans', help='transcription')
    args = parser.parse_args()

    trans = load_transcript(args.trans)
    mapping = load_mapping(args.mapping)
    trans = map_trans(trans, mapping, args.unk)
    for utt, utt_trans in trans.items():
        print(utt, ' '.join(utt_trans))


if __name__ == '__main__':
    main()

