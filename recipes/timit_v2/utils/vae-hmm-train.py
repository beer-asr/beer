'Training of the VAE-HMM model.'


import random
import argparse
import sys
import pickle
import logging
import numpy as np
import torch
import beer


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


training_types = ['viterbi', 'baum_welch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('feats', help='train features (npz file)')
    parser.add_argument('labels', type=str, help='Label file')
    parser.add_argument('vae_emissions', help='vae + emissions model')
    parser.add_argument('stats', help='stats of the training data')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--training_type', default=training_types[0],
                        choices=training_types)
    parser.add_argument('--lrate', type=float,
                        help='learning rate for the latent model')
    parser.add_argument('--lrate-nnet', type=float,
                        help='learning rate for the nnet components')
    parser.add_argument('--batch_size', type=int,
                        help='number of utterances per batch')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--use-gpu', action='store_true',
                        help='train on gpu')
    parser.add_argument('--fast-eval', action='store_true',
                        help='do not compute unecessary KL divergence term')
    args = parser.parse_args()

    # Read arguments
    feats = np.load(args.feats)
    labels = np.load(args.labels)
    hmm_mdl_dir = args.hmm_model_dir
    training_type = args.training_type
    stats = np.load(args.feat_stats)
    lrate = args.lrate
    lrate_nnet = args.lrate_nnet
    batch_size = args.batch_size
    epochs = args.epochs
    use_gpu = args.use_gpu
    fast_eval = args.fast_eval

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(args.vae_emissions, 'rb') as pickle_file:
        vae_emissions = pickle.load(pickle_file)
    vae_emissions = vae_emissions.to(device)

    # Total number of frames in the training data. This is needed to
    # compute the stochastic version of the ELBO.
    tot_counts = int(stats['nframes'])

    # Build 2 optimizers. One is for the latent model and the second
    # is for the nnet components. The latent model optimizer
    # is responsible to call the nnet optimizer.
    nnet_optim = torch.optim.Adam(vae_emissions.modules_parameters(), lr=lrate_nnet,
                                  weight_decay=1e-2)
    latent_model_optim = beer.BayesianModelCoordinateAscentOptimizer(
                                  model.mean_field_groups,
                                  lrate=lrate, std_optim=nnet_optim)

    for epoch in range(1, epochs + 1):
        logging.info("Epoch: %d", epoch)

        # At the beginning of each epoch we shuffle the order of the
        # utterances.
        keys = list(feats.keys())
        random.shuffle(keys)
        batches = [keys[i: i + batch_size]
                   for i in range(0, len(keys), batch_size)]
        logging.info("Data shuffled into %d batches", len(batches))

        for batch_keys in batches:
            # Initialize the lower-bound.
            elbo = beer.evidence_lower_bound(datasize=tot_counts)

            # Reset the gradient of the parameters.
            optimizer.zero_grad()

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
                ali_sets = beer.AlignModelSet(vae_emissions.latent_model, lab)
                hmm_ali = beer.HMM.create(init_state, final_state,
                                          trans_mat_ali, ali_sets,
                                          training_type)

                # Set the current HMM as the latent model of the
                # VAE.
                vae_emissions.latent_model = hmm_ali

                # Accumulate the ELBO for each utterance.
                elbo += beer.evidence_lower_bound(vae_emissions, ft,
                        datasize=tot_counts, fast_eval=fast_eval)

            # Compute the natural gradient for the parameters of the
            # latent model.
            elbo.natural_backward()

            # Compute the gradient of the nnet components.
            elbo.backward()

            # Update the latent model and the nnets.
            optimizer.step()

            logging.info("Elbo value is %f", float(elbo) / (tot_counts *
                         batch_nutt))

        # At the end of each epoch, output the current state of the
        # model.
        path = os.path.join(args.outdir, str(epoch) + 'mdl')
        with open(path, 'wb') as m:
            pickle.dump(vae_emissions.to(torch.device('cpu')), path)

if __name__ == "__main__":
    main()
