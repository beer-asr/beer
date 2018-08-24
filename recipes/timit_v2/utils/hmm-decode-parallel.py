'Viterbi decoding'


import argparse
import os
import pickle
import sys

import torch
import numpy as np
import beer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Decoding model')
    parser.add_argument('feats', help='data to decode')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    feats = np.load(args.feats)

    with open(args.model, 'rb') as m:
        mdl = pickle.load(m)


    for line in sys.stdin:
        utt = line.strip()
        ft = torch.from_numpy(feats[utt]).float()
        best_path = mdl.decode(ft)
        path = os.path.join(args.outdir, utt + '.npy')
        np.save(path, best_path)


if __name__ == '__main__':
    main()

