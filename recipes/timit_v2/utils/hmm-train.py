
'Train a HMM.'

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
    parser.add_argument('--alignments', help='utterance alignemnts')
    parser.add_argument('--batch-size', type=int,
                        help='utterance number in each batch')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--fast-eval', action='store_true')
    parser.add_argument('--infer-type', default='viterbi',
                        choices=['baum_welch', 'viterbi'],
                        help='how to compute the state posteriors')
    parser.add_argument('--lrate', type=float, help='learning rate')
    parser.add_argument('--tmpdir', help='directory to store intermediary ' \
                                         'models')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('hmm', help='hmm model to train')
    parser.add_argument('feats', help='Feature file')
    parser.add_argument('feat_stats', help='data statistics')
    parser.add_argument('out', help='output model')
    args = parser.parse_args()

    # Load the data for the training.
    feats = np.load(args.feats)

    ali = None
    if args.alignments:
        ali = np.load(args.alignments)

    stats = np.load(args.feat_stats)

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

    tot_counts = int(stats['nframes'])
    for epoch in range(1, args.epochs + 1):

        # Shuffle the order of the utterance.
        keys = list(feats.keys())
        random.shuffle(keys)
        batches = [keys[i: i + args.batch_size]
                   for i in range(0, len(keys), args.batch_size)]
        logging.debug('Data shuffled into {} batches'.format(len(batches)))


        # One mini-batch update.
        for batch_no, batch_keys in enumerate(batches, start=1):
            # Reset the gradients.
            optimizer.zero_grad()

            # Initialize the ELBO.
            elbo = beer.evidence_lower_bound(datasize=tot_counts)

            for utt in batch_keys:
                ft = torch.from_numpy(feats[utt]).float().to(device)

                # Get the alignment graph if provided.
                graph = None
                if ali is not None:
                    graph = ali[utt][0].to(device)

                elbo += beer.evidence_lower_bound(model, ft,
                                                  datasize=tot_counts,
                                                  fast_eval=args.fast_eval,
                                                  inference_graph=graph,
                                                  inference_type=args.infer_type)

            # Compute the gradient of the model.
            elbo.natural_backward()

            # Update the parameters.
            optimizer.step()

            elbo_value = float(elbo) / (tot_counts * len(batch_keys))
            log_msg = 'epoch={}/{}  batch={}/{}  ELBO={}'
            logging.info(log_msg.format(epoch, args.epochs,
                                        batch_no, len(batches),
                                        round(elbo_value, 3)))


        if args.tmpdir:
            path = os.path.join(args.tmpdir, str(epoch) + '.mdl')
            with open(path, 'wb') as fh:
                pickle.dump(model.to(torch.device('cpu')), fh)


    with open(args.out, 'wb') as fh:
        pickle.dump(model.to(torch.device('cpu')), fh)

if __name__ == "__main__":
    main()
