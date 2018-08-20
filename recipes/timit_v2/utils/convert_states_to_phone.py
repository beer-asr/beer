
import argparse
import sys
import logging
import numpy as np
import yaml


log_format = "%(asctime)s :%(lineno)d %(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

def read_phonelist(infile):
    dict_map = {}
    with open(infile, 'r') as p:
        for line in p:
            tokens = line.strip().split()
            dict_map[int(tokens[1])] = 0
    return dict_map

def convert_state_to_phone(state_ids, end_states, dict_state_phone):
    '''Convert state ids into phone ids
    Args:
    state_ids(list): state ids (int)
    end_states(list): ending state ids
    dict_state_phone(dict): state id (int) -> phone id (int)
    Returns:
    phone_ids(list): phone ids (int)
    '''
    phone_ids = []
    for i in range(len(state_ids) - 1):
        if (state_ids[i] != state_ids[i+1]) and (int(state_ids[i]) in end_states):
            phone_ids.append(dict_state_phone[int(state_ids[i])])
    phone_ids.append(dict_state_phone[int(state_ids[-1])])
    return phone_ids

def main():
    parser = argparse.ArgumentParser(description='Convert state ids into phone ids')
    parser.add_argument('hyp_states', help='File of decoding state ids in integer')
    parser.add_argument('hyp_phone', help='Output file of decoding phone ids in integer')
    parser.add_argument('phone_map', help='File: phones.txt')
    parser.add_argument('hmm_conf', type=str, help='Configuration file of hmm')
    args = parser.parse_args()

    hyp_states = args.hyp_states
    hyp_phone = args.hyp_phone
    dict_phone_map = read_phonelist(args.phone_map)
    hmm_conf_file = args.hmm_conf
    with open(hmm_conf_file, 'r') as f:
        hmm_conf = yaml.load(f)
    if 'n_state_per_unit' not in hmm_conf.keys():
        sys.exit('HMM configuration file keys missing: n_state_per_unit')
    n_state_per_unit = hmm_conf['n_state_per_unit']
    tot_states = list(range(len(dict_phone_map) * n_state_per_unit))

    dict_state_phone = {}
    # Create a state-id to phone-id map
    for i in dict_phone_map.keys():
        for j in range(n_state_per_unit):
            dict_state_phone[i * n_state_per_unit + j] = i

    end_states = [i + n_state_per_unit - 1 for i in dict_state_phone.keys()
                  if i % n_state_per_unit == 0]
    dict_phone_ids = {}
    with open(hyp_states, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            uttid = tokens.pop(0)
            logging.info('Processing utterance %s', uttid)
            utt_phones = convert_state_to_phone(tokens, end_states,
                         dict_state_phone)
            dict_phone_ids[uttid] = utt_phones
    np.savez(hyp_phone, **dict_phone_ids)

if __name__ == '__main__':
    main()
