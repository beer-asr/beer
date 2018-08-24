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
    parser.add_argument('--training_type', default=training_types[0],
                        choices=training_types)
    parser.add_argument('--lrate', type=float,
                        help='learning rate for the latent model')
    parser.add_argument('--lrate-nnet', type=float,
                        help='learning rate for the nnet components')
    parser.add_argument('--batch-size', type=int,
                        help='number of utterances per batch')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--use-gpu', action='store_true',
                        help='train on gpu')
    parser.add_argument('--fast-eval', action='store_true',
                        help='do not compute unecessary KL divergence term')
    parser.add_argument('--kl-weight', type=float, default=1.,
                        help='weighting of the KL divergence')
    parser.add_argument('--verbose', action='store_true',
                        help='show debug messages')
    parser.add_argument('feats', help='train features (npz file)')
    parser.add_argument('labels', type=str, help='Label file')
    parser.add_argument('vae_emissions', help='vae + emissions model')
    parser.add_argument('stats', help='stats of the training data')
    parser.add_argument('mdldir', help='output model directory')
    args = parser.parse_args()

    if args.verbose:
        logging.setLevel(logging.DEBUG)

    # Read arguments
    feats = np.load(args.feats)
    labels = np.load(args.labels)
    training_type = args.training_type
    stats = np.load(args.stats)
    mdldir = args.mdldir
    lrate = args.lrate
    lrate_nnet = args.lrate_nnet
    batch_size = args.batch_size
    epochs = args.epochs
    use_gpu = args.use_gpu
    fast_eval = args.fast_eval
    kl_weight = args.kl_weight
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    init_vae_emissions = args.vae_emissions
    final_mdl = os.path.join(mdldir, 'final.mdl')
    start_id = 1
    if not os.path.exists(final_mdl):
        for i in range(1, epochs):
            exist_mdl = os.path.join(mdldir, str(i) + '.mdl')
            if os.path.exists(exist_mdl):
                init_vae_emissions = exist_mdl
                start_id = i + 1
    with open(init_vae_emissions, 'rb') as pickle_file:
        vae_emissions = pickle.load(pickle_file)
    vae_emissions = vae_emissions.to(device)
    emissions = vae_emissions.latent_model

    # Total number of frames in the training data. This is needed to
    # compute the stochastic version of the ELBO.
    tot_counts = int(stats['nframes'])

    # Build 2 optimizers. One is for the latent model and the second
    # is for the nnet components. The latent model optimizer
    # is responsible to call the nnet optimizer.
    nnet_optim = torch.optim.Adam(vae_emissions.modules_parameters(), lr=lrate_nnet,
                                  weight_decay=1e-2)
    optim = beer.BayesianModelCoordinateAscentOptimizer(vae_emissions.mean_field_groups,
                                                        lrate=lrate,
                                                        std_optim=nnet_optim)

    for epoch in range(start_id, epochs + 1):
        logging.info("Epoch: %d", epoch)

        # At the beginning of each epoch we shuffle the order of the
        # utterances.
        keys = list(feats.keys())
        random.shuffle(keys)
        batches = [keys[i: i + batch_size]
                   for i in range(0, len(keys), batch_size)]
        logging.debug("Data shuffled into %d batches", len(batches))

        for batch_keys in batches:
            # Initialize the lower-bound.
            elbo = beer.evidence_lower_bound(datasize=tot_counts)

            # Reset the gradient of the parameters.
            optim.zero_grad()

            batch_nutt = len(batch_keys)
            for utt in batch_keys:
                logging.debug("processing utterance %s", utt)

                # Load the features and the labels.
                ft = torch.from_numpy(feats[utt]).float().to(device)
                lab = labels[utt]

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

                # Accumulate the ELBO for each utterance.
                elbo += beer.evidence_lower_bound(vae_emissions, ft,
                        datasize=tot_counts,
                        kl_weight=kl_weight,
                        inference_type=training_type, fast_eval=fast_eval)

            # Compute the natural gradient for the parameters of the
            # latent model.
            elbo.natural_backward()

            # Compute the gradient of the nnet components.
            elbo.backward()

            # Update the latent model and the nnets.
            optim.step()

            elbo_value = float(elbo) / (tot_counts * batch_nutt)
            logging.info("ln p(X) >= {}".format(round(elbo_value, 3)))

        # At the end of each epoch, output the current state of the
        # model.
        mdlname = os.path.join(mdldir, str(epoch) + '.mdl')
        vae_emissions.latent_model = emissions
        if epoch != epochs:
            with open(mdlname, 'wb') as m:
                pickle.dump(vae_emissions.to(torch.device('cpu')), m)
    with open(final_mdl, 'wb') as m:
        pickle.dump(vae_emissions.to(torch.device('cpu')), m)

if __name__ == "__main__":
    main()
