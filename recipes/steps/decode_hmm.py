

import torch
import pickle
import numpy as np
import sys
sys.path.insert(0, '../../beer')
import beer
import argparse
import funcs
import logging


log_format = "%(asctime)s :%(lineno)d %(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

def filter_text(old_text, dict_map=None, remove_sys = None):
    '''Convert text into new phoneme set mapping, i.e, from 48 phonemes
        set to 39 phonemes set; remove certain symbols
    Args:
        old_text(list): list of characters(str)
        dict_map(dict): original phone(str) -> new phone(str)
        remove_sys(str): symbol to be removed.
    Return:
        new_text(list): list of filtered characters(str)
    '''
    if dict_map is not None:
        if remove_sys is not None:
            new_text = [dict_map[i] for i in old_text if i != remove_sys]
        else:
            new_text = [dict_map[i] for i in old_text]
    else:
        if remove_sys is not None:
            new_text = [i for i in old_text if i != remove_sys]
        else:
            new_text = old_text
    return new_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Decoding model')
    parser.add_argument('decode_dir', type=str, help='Decoding directory')
    parser.add_argument('feats', type=str, help='Features to decode')
    parser.add_argument('transcription', type=str, help='True transcription')
    parser.add_argument('phonefile', type=str, help='List of phonemes(defaul 48)')
    parser.add_argument('nstate_per_phone', type=int)
    parser.add_argument('--gamma', type=float, default=.5,
        help='Probability to jump to new phone in transition matrix')
    parser.add_argument('--phone_39', type=str,
        help='Use 39 phonemes set')
    parser.add_argument('--remove_sys', type=str,
        help='Phoneme to be removed during scoring')
    parser.add_argument('--score', action='store_true')
    #parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    feats = np.load(args.feats)
    decode_dir = args.decode_dir
    trans = args.transcription
    phonefile = args.phonefile
    nstate_per_phone = args.nstate_per_phone
    gamma = args.gamma
    phone_39_map = args.phone_39
    remove_sys = args.remove_sys
    score = args.score
    decode_results = decode_dir + ('/decode_results.txt')

    with open(args.model, 'rb') as m:
        mdl = pickle.load(m)

    #use_gpu = args.use_gpu
    #if use_gpu:
    #    device = torch.device('cuda')
    #else:
    #    device = torch.device('cpu')


    dict_phone_ids, dict_state_to_phone = funcs.create_phone_dict(phonefile,
                                          nstate_per_phone)
    if phone_39_map != "":
        dict_phone_39_map = funcs.create_48_39_phone_map(phone_39_map)
    else:
        dict_phone_39_map = None
    if remove_sys == "":
        remove_sys == None

    init_states = []
    final_states = []
    for i in dict_state_to_phone.keys():
        if i % nstate_per_phone == 0:
            init_states.append(i)
            final_states.append(i + nstate_per_phone - 1)

    unit_priors = torch.ones(len(dict_phone_ids)) / len(dict_phone_ids)
    trans_mat = beer.HMM.create_trans_mat(unit_priors, nstate_per_phone, gamma)
    hmm = beer.HMM.create(init_states, final_states, trans_mat, mdl,
                          training_type='viterbi')
    dict_trans = funcs.read_transcription(trans)
    dict_hyps = {}

    with open(decode_results, 'w') as f:
        for k in feats.keys():
            logging.info('Decoding utt %s', k)
            if score:
                dict_trans[k] = filter_text(dict_trans[k], dict_phone_39_map,
                                remove_sys)
            ft = torch.from_numpy(feats[k]).float()
            best_path = hmm.decode(ft)
            hyp_phones = funcs.convert_state_to_phone(dict_state_to_phone,
                         list(best_path.numpy()), nstate_per_phone)
            hyp_phones = filter_text(hyp_phones, dict_phone_39_map, remove_sys)
            dict_hyps[k] = hyp_phones
            f.write(k + ' ' + ' '.join(hyp_phones) + '\n')

    if score:
        logging.info('Scoring') 
        print('Error rate is ' + '{0:.4f}'.format(funcs.score(dict_trans, dict_hyps)))

if __name__ == '__main__':
    main()
