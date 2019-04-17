
'Compute the recall/precision/fscore of the units boundaries.'


import argparse

import numpy as np
import beer

def load_transcript(path):
    with open(path, 'r') as f:
        trans = {}
        for line in f:
            tokens = line.strip().split()
            trans[tokens[0]] = tokens[1:]
    return trans

def boundaries(utt):
    boundaries = []
    frame = 1
    current_unit = utt[0]
    for frame, unit in enumerate(utt[1:], start=2):
        if unit != current_unit:
            current_unit = unit
            boundaries.append(frame)
    return np.array(boundaries)


def get_hits(ref_bounds, hyp_bounds, delta=1):
    hits = 0
    for b in ref_bounds:
        min_dist = np.abs(b - hyp_bounds).min()
        if min_dist <= delta:
            hits += 1
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delta', default=1, type=int,
                        help='acceptance threshold (in frame) to consider a ' \
                             'boundary a "hit"')
    parser.add_argument('ref', help='reference frame transcription')
    parser.add_argument('hyp', help='hypothesis frame transcription')
    args = parser.parse_args()

    ref = load_transcript(args.ref)
    hyp = load_transcript(args.hyp)

    hits = 0
    misses = 0
    ref_bounds_count = 0
    hyp_bounds_count = 0
    for utt in ref:
        ref_bounds = boundaries(ref[utt])
        hyp_bounds = boundaries(hyp[utt])
        hits += get_hits(ref_bounds, hyp_bounds, delta=args.delta)
        misses += len(ref_bounds) - hits
        ref_bounds_count += len(ref_bounds)
        hyp_bounds_count += len(hyp_bounds)

    recall = hits / ref_bounds_count
    prec = hits / hyp_bounds_count
    fscore = 2 * (prec * recall) / (prec + recall)

    print(f'recall {100 * recall:.2f} precision {100 * prec:.2f} fscore:{100 * fscore: .2f}')


if __name__ == "__main__":
    main()

