
'Train a HMM with a given alignments.'

import random
import numpy as np
import torch
import argparse
import sys
import beer
import pickle
import logging
import os

log_format = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def load_batch(feats, alis, keys):
    labels = torch.cat([torch.from_numpy(alis[key]).long()
                        for key in keys])
    ft = torch.cat([torch.from_numpy(feats[key]).float()
                    for key in keys])
    return ft, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='utterance number in each batch')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--fast-eval', action='store_true')
    parser.add_argument('--kl-weight', type=float, default=1.,
                        help='weighting of KL div. of the ELBO')
    parser.add_argument('--lrate-nnet', type=float, default=1e-3,
                        help='learning rate for the nnet components')
    parser.add_argument('--lrate', type=float, default=1.,
                        help='learning rate')
    parser.add_argument('--nnet-optim-state',
                        help='file where to load/save state of the nnet '
                             'optimizer')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('model', help='model to train')
    parser.add_argument('alis', help='alignments')
    parser.add_argument('feats', help='Feature file')
    parser.add_argument('feat_stats', help='data statistics')
    parser.add_argument('out', help='output model')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load the data.
    alis = np.load(args.alis)
    feats = np.load(args.feats)
    stats = np.load(args.feat_stats)

    # Load the model and move it to the chosen device (CPU/GPU)
    with open(args.model, 'rb') as fh:
        model = pickle.load(fh)
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # NNET optimizer.
    nnet_optim = torch.optim.Adam(
        model.modules_parameters(),
        lr=args.lrate_nnet,
        weight_decay=1e-2
    )

    if args.nnet_optim_state and os.path.isfile(args.nnet_optim_state):
        logging.debug('load nnet optimizer state: {}'.format(args.nnet_optim_state))
        optim_state = torch.load(args.nnet_optim_state)
        nnet_optim.load_state_dict(optim_state)

    # Prepare the optimizer for the training.
    params = model.mean_field_groups
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(params,
                                                            lrate=args.lrate,
                                                            std_optim=nnet_optim)

    # If no batch_size is specified, use the whole data.
    batch_size = len(feats.files)
    if args.batch_size > 0:
        batch_size = args.batch_size


    tot_counts = int(stats['nframes'])
    for epoch in range(args.epochs):
        # Shuffle the order of the utterance.
        keys = list(feats.keys())
        random.shuffle(keys)
        batches = [keys[i: i + batch_size]
                   for i in range(0, len(keys), batch_size)]
        logging.debug('Data shuffled into {} batches'.format(len(batches)))

        for batch_no, batch_keys in enumerate(batches, start=1):
            # Reset the gradients.
            optimizer.zero_grad()

            # Load the batch data.
            ft, labels = load_batch(feats, alis, batch_keys)
            ft, labels = ft.to(device), labels.to(device)

            # Compute the objective function.
            elbo = beer.evidence_lower_bound(model, ft, state_path=labels,
                                             kl_weight=args.kl_weight,
                                             datasize=tot_counts,
                                             fast_eval=args.fast_eval)

            # Compute the gradient of the model.
            elbo.natural_backward()
            elbo.backward()

            # Update the parameters.
            optimizer.step()

            elbo_value = float(elbo) / tot_counts
            log_msg = 'epoch={}/{} batch={}/{} elbo={}'
            logging.info(log_msg.format(
                epoch, args.epochs,
                batch_no, len(batches),
                round(elbo_value, 3))
            )


    if args.nnet_optim_state:
        torch.save(nnet_optim.state_dict(), args.nnet_optim_state)

    with open(args.out, 'wb') as fh:
        pickle.dump(model.to(torch.device('cpu')), fh)


if __name__ == "__main__":
    main()
