

import torch
import pickle
import numpy as np
import sys
sys.path.insert(0, '../../beer')
from feature_transform import feature_transform
import beer
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Dedoding HMM model')
    parser.add_argument('decode_dir', type=str, help='Decoding directory')
    parser.add_argument('feats', type=str, help='Features to decode')
    parser.add_argument('feat_stats', type=str, help='Feature statistics')
    parser.add_argument('phonefile', type=str)
    parser.add_argument('nstate_per_phone', type=int)
    parser.add_argument('--mean_normalize', default='True',
        choices=['True', 'False'], help='Should be same as in training setup')
    parser.add_argument('--var_normalize', default='False',
        choices=['True', 'False'], help='Should be same as in training setup')
    parser.add_argument('--context', type=int, default=5,
        help='Should be same as in training setup')
    parser.add_argument('--gamma', type=float, default=.5, 
        help='Probability to jump to new phone in transition matrix')
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    feats = np.load(args.feats)
    decode_dir = args.decode_dir
    feat_stats = np.load(args.feat_stats)
    phonefile = args.phonefile
    nstate_per_phone = args.nstate_per_phone
    mean_norm = args.mean_normalize
    var_norm = args.var_normalize
    context = args.context
    gamma = args.gamma
    with open(args.model, 'rb') as m:
        mdl = pickle.load(m)

    use_gpu = args.use_gpu
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    phonelist, dict_phone_to_state = create_phone_dict(phonefile,
                                                       nstate_per_phone)
    global_mean = feat_stats['mean']
    global_std = feat_stats['std']

    init_states = []
    final_states = []
    for i in dict_phone_to_state.keys():
        if i % nstate_per_phone == 0:
            init_states.append(i)
            final_states.append(i + nstate_per_phone - 1)

    unit_priors = torch.distributions.Dirichlet(torch.ones(len(phonelist))).sample()
    trans_mat = beer.HMM.create_trans_mat(unit_priors, nstate_per_phone, gamma)
    hmm = beer.HMM.create(init_states, final_states, trans_mat, mdl,
                          training_type='viterbi')
    
    for k in feats.keys():
        ft = feature_transform(feats[k], mean_norm=mean_norm,
                var_norm=var_norm, mean=global_mean, std=global_std,
                context=context)
        best_path = hmm.decode(torch.from_numpy(ft).float().to(device))
        best_path = np.asarray([])

