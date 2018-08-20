
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
    parser.add_argument('--infer_type', type=str, default='viterbi',
        choices=['baum_welch', 'viterbi'], help='Method to compute elbo value')
    parser.add_argument('--lrate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int,
        help='Utterance number in each batch')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--fast-eval', action='store_true')
    parser.add_argument('feats', type=str, help='Feature file')
    parser.add_argument('labels', type=str, help='States label file')
    parser.add_argument('emissions', type=str, help='Emission modelset file')
    parser.add_argument('feat_stats', type=str, default=None,
        help='Feature statistics: mean, standard deviation, count')
    parser.add_argument('hmm_model_dir', type=str, help='Output trained HMM model')
    args = parser.parse_args()

    feats = np.load(args.feats)
    labels = np.load(args.labels)
    hmm_mdl_dir = args.hmm_model_dir
    infer_type = args.infer_type
    stats = np.load(args.feat_stats)
    lrate = args.lrate
    batch_size = args.batch_size
    epochs = args.epochs
    use_gpu = args.use_gpu
    fast_eval = args.fast_eval

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    init_mdl = args.emissions
    final_mdl = os.path.join(hmm_mdl_dir, 'final.mdl')
    start_id = 1
    if not os.path.exists(final_mdl):
        for i in range(1, epochs):
            exist_mdl = os.path.join(hmm_mdl_dir, str(i) + '.mdl')
            if os.path.exists(exist_mdl):
                init_mdl = exist_mdl
                start_id = i + 1
    with open(init_mdl, 'rb') as pickle_file:
        emissions = pickle.load(pickle_file)
    emissions = emissions.to(device)
    tot_counts = int(stats['nframes'])

    params = emissions.mean_field_groups
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(params, lrate=lrate)

    for epoch in range(start_id, epochs + 1):
        logging.info("Epoch: %d", epoch)
        keys = list(feats.keys())
        random.shuffle(keys)
        batches = [keys[i: i+batch_size] for i in range(0, len(keys), batch_size)]
        logging.info("Data shuffled into %d batches", len(batches))
        hmm_epoch = os.path.join(hmm_mdl_dir, str(epoch) + '.mdl')
        for batch_keys in batches:
            optimizer.zero_grad()
            elbo = beer.evidence_lower_bound(datasize=tot_counts,
                fast_eval=fast_eval)
            batch_nutt = len(batch_keys)
            for utt in batch_keys:
                logging.info("Training with utterance %s", utt)
                ft = torch.from_numpy(feats[utt]).float().to(device)
                lab = labels[utt]
                init_state = torch.tensor([0]).to(device)
                final_state = torch.tensor([len(lab) - 1]).to(device)
                trans_mat_ali = beer.HMM.create_ali_trans_mat(len(lab)).to(device)
                ali_sets = beer.AlignModelSet(emissions, lab)
                hmm_ali = beer.HMM.create(init_state, final_state,
                          trans_mat_ali, ali_sets)
                elbo += beer.evidence_lower_bound(hmm_ali, ft,
                        datasize=tot_counts, fast_eval=fast_eval,
                        inference_type=infer_type)
            elbo.natural_backward()
            logging.info("Elbo value is %f", float(elbo) / (tot_counts *
                         batch_nutt))
            optimizer.step()
        if epoch != epochs:
            with open(hmm_epoch, 'wb') as m:
                pickle.dump(emissions.to(torch.device('cpu')), m)
    with open(final_mdl, 'wb') as m:
        pickle.dump(emissions.to(torch.device('cpu')), m)

if __name__ == "__main__":
    main()
