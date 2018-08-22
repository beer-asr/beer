
import sys
import os
import argparse
import logging
import pickle
import yaml
import torch
import numpy as np
import beer


log_format = "%(asctime)s :%(lineno)d %(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Decoding model')
    parser.add_argument('feats', type=str, help='data to decode')
    args = parser.parse_args()

    feats = np.load(args.feats)

    with open(args.model, 'rb') as m:
        mdl = pickle.load(m)

    for utt in feats.keys():
        logging.debug('Decoding utt {}'.format(utt))
        ft = torch.from_numpy(feats[utt]).float()
        best_path = mdl.decode(ft)
        best_path = [str(int(v)) for v in best_path]
        print(utt, ' '.join(best_path))

if __name__ == '__main__':
    main()

