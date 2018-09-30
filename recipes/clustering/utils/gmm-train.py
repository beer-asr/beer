
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--fast-eval', action='store_true')
    parser.add_argument('--lrate', type=float, default=1.,
                        help='learning rate')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('model', help='model to train')
    parser.add_argument('batches', help='list of batches file')
    parser.add_argument('feat_stats', help='data statistics')
    parser.add_argument('out', help='output model')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load the data.
    stats = np.load(args.feat_stats)

    # Load the batches.
    batches_list = []
    with open(args.batches, 'r') as f:
        for line in f:
            batches_list.append(line.strip())

    # Load the model and move it to the chosen device (CPU/GPU)
    with open(args.model, 'rb') as fh:
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


    tot_counts = int(stats['nframes'])
    for epoch in range(1, args.epochs + 1):
        # Shuffle the order of the utterance.
        random.shuffle(batches_list)
        for batch_no, path in enumerate(batches_list, start=1):
            # Reset the gradients.
            optimizer.zero_grad()

            # Load the batch data.
            batch = np.load(path)
            ft = torch.from_numpy(batch['features']).float()
            ft = ft.to(device)

            # Compute the objective function.
            elbo = beer.evidence_lower_bound(model, ft, datasize=tot_counts,
                                             fast_eval=args.fast_eval)

            # Compute the gradient of the model.
            elbo.natural_backward()

            # Update the parameters.
            optimizer.step()

            elbo_value = float(elbo) / tot_counts
            log_msg = 'epoch={}/{} batch={}/{} elbo={}'
            logging.info(log_msg.format(
                epoch, args.epochs,
                batch_no, len(batches_list),
                round(elbo_value, 3))
            )

    with open(args.out, 'wb') as fh:
        pickle.dump(model.to(torch.device('cpu')), fh)


if __name__ == "__main__":
    main()
