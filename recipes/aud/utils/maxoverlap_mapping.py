'''Create a unit->phone mapping given two framewise transcriptions
(i.e. aligments)

The mapping is created by assigning to a unit the phone it overlap the
most with.

'''

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', help='file to output the total frame counts')
    parser.add_argument('ref', help='reference alignments')
    parser.add_argument('hyp', help='proposed alignments')
    parser.add_argument('mapping', help='Output mapping')
    args = parser.parse_args()

    ref_align = load_transcript(args.ref)
    hyp_align = load_transcript(args.hyp)

    counts = defaultdict(lambda: defaultdict(int))
    for utt  in ref_align:
        for ref_unit, hyp_unit in zip(ref_align[utt], hyp_align[utt]):
            counts[hyp_unit][ref_unit] += 1

    if args.counts:
        dcounts = {}
        for unit, ucounts in counts.items():
            dcounts[unit] = {}
            for phone, count in ucounts.items():
                dcounts[unit][phone] = count
        with open(args.counts, 'w') as f:
            yaml.dump(dcounts, f, default_flow_style=False)

    mapping = {au: max(label_counts, key=label_counts.get)
               for au, label_counts in counts.items()}

    with open(args.mapping, 'w') as f:
        for key, val in mapping.items():
            print(key, val, file=f)


if __name__ == '__main__':
    main()

