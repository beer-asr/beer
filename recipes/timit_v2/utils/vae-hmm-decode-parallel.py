
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
    parser.add_argument('model', help='model to decode with')
    parser.add_argument('feats', help='Feature file')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    # Load the data for the training.
    feats = np.load(args.feats)

    with open(args.model, 'rb') as fh:
        model = pickle.load(fh)

    for line in sys.stdin:
        uttid = line.strip()
        ft = torch.from_numpy(feats[uttid]).float()
        enc_states = model.encoder(ft)
        post_params = model.encoder_problayer(enc_states)
        samples, _ = model.encoder_problayer.samples_and_llh(post_params,
                                                             use_mean=True)
        best_path = model.latent_model.decode(samples)
        path = os.path.join(args.outdir, uttid + '.npy')
        np.save(path, best_path)


if __name__ == "__main__":
    main()
