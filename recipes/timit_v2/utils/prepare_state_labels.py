
import argparse
import numpy as np
import os
import yaml
import logging

logging.basicConfig(format='%(levelname)s: %(message)s')
n_state_per_unit = 3

def read_phonelist(infile):
    dict_map = {}
    with open(infile, 'r') as p:
        for line in p:
            tokens = line.strip().split()
            dict_map[int(tokens[1])] = 0
    return dict_map

def main():
    parser = argparse.ArgumentParser(description='Convert phone integer\
        transcriptions into hmm states integer labels, used for HMM training')
    parser.add_argument('phonefile', help='phones.txt')
    parser.add_argument('phoneids', help='phone.int.npz file')
    parser.add_argument('conf', help='HMM configuration file')
    parser.add_argument('outdir', help='Output directory')
    args = parser.parse_args()

    dict_phone = read_phonelist(args.phonefile)
    text_npz = np.load(args.phoneids)
    outfile = os.path.join(args.outdir, 'states.int.npz')
    with open(args.conf, 'r') as f:
        conf = yaml.load(f)
    if 'n_state_per_unit' not in conf.keys():
        logging.error('n_state_per_unit not in conf file')
        exit(1)
    n_state_per_unit = conf['n_state_per_unit']

    tot_states = list(range(len(dict_phone) * n_state_per_unit))
    dict_state_phone = dict((s, tot_states[s*n_state_per_unit:(s+1)*n_state_per_unit])
                    for s in dict_phone.keys())
    dict_states = {}
    utt_keys = list(text_npz.keys())
    for k in utt_keys:
        utt = text_npz[k]
        states = [dict_state_phone[s] for s in utt]
        states = np.array(states).reshape(-1)
        dict_states[k] = states
    np.savez(outfile, **dict_states)

if __name__ == "__main__":
    main()
