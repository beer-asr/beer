
import sys
import os
import argparse
import logging
import pickle
import yaml
import torch
import numpy as np
import beer


log_format = "%(asctime)s :%(lineno)d %(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Decoding model')
    parser.add_argument('decode_dir', type=str, help='Decoding directory')
    parser.add_argument('feats', type=str, help='Features to decode')
    parser.add_argument('hmm_conf', type=str, help='Configuration file of hmm')
    parser.add_argument('--gamma', type=float, default=.5,
        help='Probability to jump to new phone in transition matrix')
    #parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    feats = np.load(args.feats)
    decode_dir = args.decode_dir
    gamma = args.gamma
    decode_results = os.path.join(decode_dir,'decode_results.txt')
    with open(args.model, 'rb') as m:
        mdl = pickle.load(m)
    with open(args.hmm_conf, 'r') as conf:
        hmm_conf = yaml.load(conf)

    if 'n_units' not in hmm_conf.keys() or 'n_state_per_unit' not in hmm_conf.keys():
        sys.exit('HMM configuration file keys missing: n_units or n_state_per_unit')
    n_units = hmm_conf['n_units']
    n_state_per_unit = hmm_conf['n_state_per_unit']

    init_states = []
    final_states = []
    for i in range(n_units * n_state_per_unit):
        if i % n_state_per_unit == 0:
            init_states.append(i)
            final_states.append(i + n_state_per_unit - 1)
    unit_priors = torch.ones(n_units) / n_units
    trans_mat = beer.HMM.create_trans_mat(unit_priors, n_state_per_unit, gamma)
    hmm = beer.HMM.create(init_states, final_states, trans_mat, mdl)
    
    with open(decode_results, 'w') as f:
        for k in feats.keys():
            logging.info('Decoding utt %s', k)
            ft = torch.from_numpy(feats[k]).float()
            best_path = hmm.decode(ft)
            best_path = [str(int(v)) for v in best_path]
            print(k, ' '.join(best_path), file=f)

if __name__ == '__main__':
    main()
