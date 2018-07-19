
''' HMM model training
'''
import random
import numpy as np
import torch
import argparse
import sys
sys.path.insert(0, '../../beer')
import beer
import pickle
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('feats', type=str, help='Feature file')
    parser.add_argument('labels', type=str, help='Label file')
    parser.add_argument('emissions', type=str, help='Emission modelset file')
    parser.add_argument('feat_stats', type=str, default=None,
        help='Feature statistics: mean, standard deviation, count')
    parser.add_argument('hmm_model_dir', type=str, help='Output trained HMM model')
    parser.add_argument('--training_type', type=str, default='viterbi',
        choices=['baum_welch', 'viterbi'])
    parser.add_argument('--lrate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int,
        help='Utterance number in each batch')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()



    # Read arguments
    feats = np.load(args.feats)
    labels = np.load(args.labels)
    hmm_mdl_dir = args.hmm_model_dir
    training_type = args.training_type
    stats = np.load(args.feat_stats)
    lrate = args.lrate
    batch_size = args.batch_size
    epochs = args.epochs
    use_gpu = args.use_gpu

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(args.emissions, 'rb') as pickle_file:
        emissions = pickle.load(pickle_file)
    emissions.to(device)

    tot_counts = int(stats['nframes'])

    log_format = "%(asctime)s :%(lineno)d %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)



    # Training
    params = emissions.grouped_parameters
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(*params, lrate=lrate)

    for epoch in range(epochs):
        logging.info("Epoch: %d", epoch)
        # Prepare data
        keys = list(feats.keys())
        random.shuffle(keys)
        batches = [keys[i: i+batch_size] for i in range(0, len(keys), batch_size)]
        logging.info("Data shuffled into %d batches", len(batches))
 
        hmm_epoch = hmm_mdl_dir + '/' + str(epoch) + '.mdl'
        for batch_keys in batches:
            optimizer.zero_grad()
            elbo = beer.evidence_lower_bound(datasize=tot_counts)
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
                          trans_mat_ali, ali_sets, training_type)
                #hmm_ali = beer.HMM.create(init_state, final_state,
                #          trans_mat_ali, ali_sets, training_type).to(device)
                elbo += beer.evidence_lower_bound(hmm_ali, ft,
                        datasize=tot_counts)
            elbo.natural_backward()
            logging.info("Elbo value is %f", float(elbo) / (tot_counts *
                         batch_nutt))
            optimizer.step()
        with open(hmm_epoch, 'wb') as m:
            pickle.dump(emissions.to(torch.device('cpu')), m)

    hmm_mdl = hmm_mdl_dir + '/hmm.mdl'
    with open(hmm_mdl, 'wb') as m:
        pickle.dump(emissions.to(torch.device('cpu')), m)

if __name__ == "__main__":
    main()
