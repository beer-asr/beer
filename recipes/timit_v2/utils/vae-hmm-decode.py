
'Estimate the state alignments.'

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch

import beer


log_format = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hmm', help='hmm model to train')
    parser.add_argument('feats', help='Feature file')
    args = parser.parse_args()

    # Load the data for the training.
    feats = np.load(args.feats)

    with open(args.hmm, 'rb') as fh:
        model = pickle.load(fh)

    for uttid in feats.keys():
        ft = torch.from_numpy(feats[uttid]).float()
        enc_states = model.encoder(ft)
        post_params = model.encoder_problayer(enc_states)
        samples, _ = model.encoder_problayer.samples_and_llh(post_params,
                                                             use_mean=True)
        best_path = model.latent_model.decode(samples)
        best_path = [str(int(v)) for v in best_path]
        print(uttid, ' '.join(best_path))


if __name__ == "__main__":
    main()
