'Training of the VAE-HMM model.'


import random
import argparse
import sys
import pickle
import logging
import os
import numpy as np
import torch
import beer


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


training_types = ['viterbi', 'baum_welch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', action='store_true',
                        help='train on gpu')
    parser.add_argument('--verbose', action='store_true',
                        help='show debug messages')
    parser.add_argument('feats', help='train features (npz file)')
    parser.add_argument('vae_emissions', help='vae + emissions model')
    parser.add_argument('stats', help='stats of the training data')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    if args.verbose:
        logging.setLevel(logging.DEBUG)

    # Read arguments
    feats = np.load(args.feats)
    stats = np.load(args.stats)
    use_gpu = args.use_gpu

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(args.vae_emissions, 'rb') as pickle_file:
        vae_emissions = pickle.load(pickle_file)
    vae_emissions = vae_emissions.to(device)
    emissions = vae_emissions.latent_model


    # Create the HMM for decoding.
    unit_priors = torch.ones(len(dict_phone_ids)) / len(dict_phone_ids)
    trans_mat = beer.HMM.create_trans_mat(unit_priors, nstate_per_phone, gamma)
    hmm = beer.HMM.create(init_states, final_states, trans_mat, mdl,
                          training_type='viterbi')


    # At the beginning of each epoch we shuffle the order of the
    # utterances.
    keys = list(feats.keys())
    random.shuffle(keys)
    batches = [keys[i: i + batch_size]
               for i in range(0, len(keys), batch_size)]
    logging.debug("Data shuffled into %d batches", len(batches))

    for utt in feats:
        logging.debug("decoding utterance %s", utt)

        # Load the features and the labels.
        ft = torch.from_numpy(feats[utt]).float().to(device)

        # We create the HMM structure for the forced alignment
        # on the fly.
        init_state = torch.tensor([0]).to(device)
        final_state = torch.tensor([len(lab) - 1]).to(device)
        trans_mat_ali = beer.HMM.create_ali_trans_mat(len(lab)).to(device)
        ali_sets = beer.AlignModelSet(emissions, lab)
        hmm_ali = beer.HMM.create(init_state, final_state,
                                  trans_mat_ali, ali_sets)

        # Set the current HMM as the latent model of the
        # VAE.
        vae_emissions.latent_model = hmm_ali


if __name__ == "__main__":
    main()
