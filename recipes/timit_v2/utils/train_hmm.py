
''' HMM model training
'''
import random
import numpy as np
import torch
import argparse
import sys
import beer
import pickle
import logging
import os

log_format = "%(asctime)s :%(lineno)d %(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer-type', default='viterbi',
                        choices=['baum_welch', 'viterbi'],
                        help='how to compute the state posteriors')
    parser.add_argument('--lrate', type=float, help='learning rate')
    parser.add_argument('--batch-size', type=int,
                        help='utterance number in each batch')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--fast-eval', action='store_true')
    parser.add_argument('feats', type=str, help='Feature file')
    parser.add_argument('ali', type=str, help='alignments')
    parser.add_argument('hmm', type=str, help='hmm model to train')
    parser.add_argument('feat_stats', help='data statistics')
    parser.add_argument('hmm_model_dir', type=str, help='Output trained HMM model')
    args = parser.parse_args()

    feats = np.load(args.feats)
    ali = np.load(args.ali)
    hmm_mdl_dir = args.hmm_model_dir
    stats = np.load(args.feat_stats)

    init_mdl = args.hmm
    final_mdl = os.path.join(hmm_mdl_dir, 'final.mdl')
    start_id = 1
    if not os.path.exists(final_mdl):
        for i in range(1, args.epochs):
            exist_mdl = os.path.join(hmm_mdl_dir, str(i) + '.mdl')
            if os.path.exists(exist_mdl):
                init_mdl = exist_mdl
                start_id = i + 1

    with open(args.hmm, 'rb') as fh:
        model = pickle.load(fh)

    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    params = model.mean_field_groups
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(params,
                                                            lrate=args.lrate)
    tot_counts = int(stats['nframes'])

    for epoch in range(start_id, args.epochs + 1):
        logging.info("Epoch: {}".format(epoch))

        keys = list(feats.keys())
        random.shuffle(keys)
        batches = [keys[i: i + args.batch_size]
                   for i in range(0, len(keys), args.batch_size)]
        logging.info("Data shuffled into %d batches", len(batches))

        hmm_epoch = os.path.join(hmm_mdl_dir, str(epoch) + '.mdl')

        for batch_keys in batches:
            optimizer.zero_grad()
            elbo = beer.evidence_lower_bound(datasize=tot_counts)
            batch_nutt = len(batch_keys)
            for utt in batch_keys:
                logging.info("Training with utterance %s", utt)
                ft = torch.from_numpy(feats[utt]).float().to(device)
                graph = ali[utt][0].to(device)
                telbo = beer.evidence_lower_bound(model, ft,
                                                  datasize=tot_counts,
                                                  fast_eval=args.fast_eval,
                                                  inference_graph=graph,
                                                  inference_type=args.infer_type)
                elbo += telbo
            elbo.natural_backward()

            logging.info("Elbo value is %f", float(elbo) / (tot_counts *
                         batch_nutt))
            optimizer.step()

        if epoch != args.epochs:
            with open(hmm_epoch, 'wb') as m:
                pickle.dump(model.to(torch.device('cpu')), m)

    with open(final_mdl, 'wb') as m:
        pickle.dump(model.to(torch.device('cpu')), m)

if __name__ == "__main__":
    main()
