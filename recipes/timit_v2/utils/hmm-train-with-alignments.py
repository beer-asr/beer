
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
    parser.add_argument('--fast-eval', action='store_true')
    parser.add_argument('--lrate', type=float, default=1.,
                        help='learning rate')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('hmm', help='hmm model to train')
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
    with open(args.hmm, 'rb') as fh:
        model = pickle.load(fh)
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # Prepare the optimizer for the training.
    params = model.mean_field_groups
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(params,
                                                            lrate=args.lrate)

    # If no batch_size is specified, use the whole data.
    batch_size = len(feats.files)
    if args.batch_size > 0:
        batch_size = args.batch_size


    tot_counts = int(stats['nframes'])

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
                                         datasize=tot_counts,
                                         fast_eval=args.fast_eval)

        # Compute the gradient of the model.
        elbo.natural_backward()

        # Update the parameters.
        optimizer.step()

        elbo_value = float(elbo) / tot_counts
        log_msg = 'Evidence Lower Bound: {}'
        logging.info(log_msg.format(round(elbo_value, 3)))


    with open(args.out, 'wb') as fh:
        pickle.dump(model.to(torch.device('cpu')), fh)

if __name__ == "__main__":
    main()
