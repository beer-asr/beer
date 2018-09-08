
'''Given a set of features and the corresponding alignments,
extract the features frame corresponding to the center of each
token in the alignments.'''

import argparse
import glob
import os
import sys

import numpy as np

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ali', help='alignments')
    parser.add_argument('out', help='features directory')
    args = parser.parse_args()

    # List of phones to exclude.
    to_exclude = [phone for phone in args.exclude.split(',')]

    # Create the phone <-> id mapping.
    phones2id = {}
    count = 0
    with open(args.phones, 'r') as fid:
        for line in fid:
            tokens = line.strip().split()
            if tokens[0] not in to_exclude:
                phones2id[tokens[0]] = count
                count += 1

    # Keys to extracts.
    with open(args.keys, 'r') as fid:
        keys = [line.strip() for line in fid]

    # Load the alignments.
    mlf = asrio.read_mlf(args.mlf)

    # Load the features.
    arrays = np.load(args.infile)

    labels = []
    features = []
    for uttid in arrays:
        fea = arrays[uttid]
        try:
            ali = mlf[uttid]
        except KeyError:
            print('[warning]: no alignment for utterance:', uttid, file=sys.stderr)
            continue
        for entry in ali:
            if not entry[0] in to_exclude:
                #phoneid = phones2id[entry[0]]
                #center = int(entry[1] + .5 * (entry[2] - entry[1]))
                #features.append(fea[center]), labels.append(phoneid)
                phoneid = phones2id[entry[0]]
                center = int(entry[1] + .5 * (entry[2] - entry[1]))
                features.append(fea[center]), labels.append(phoneid)
                for n in range(1, args.add_extra_frame + 1):
                    try:
                        fea_past = fea[center - n]
                        fea_future = fea[center + n]
                        features.append(fea_past), labels.append(phoneid)
                        features.append(fea_future), labels.append(phoneid)
                    except IndexError:
                        pass


    # Derive the name of the output file from the input file.
    bname = os.path.basename(args.infile)
    root, _ = os.path.splitext(bname)
    outpath = os.path.join(args.outdir, root)
    np.savez_compressed(outpath, features=features, labels=labels)


if __name__ == '__main__':
    run()

