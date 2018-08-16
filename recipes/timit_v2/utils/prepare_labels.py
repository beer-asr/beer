
import argparse
import numpy as np

def read_phonelist(infile):
    dict_map = {}
    with open(infile, 'r') as p:
        for line in p:
            tokens = line.strip().split()
            dict_map[int(tokens[1])] = 0
    return dict_map

def main():
    parser = argparse.ArgumentParser(description='Convert phone ids into states ids')
    parser.add_argument('phonefile', help='phones.txt')
    parser.add_argument('phoneids', help='phone.int.npz file')
    parser.add_argument('nstate_per_phone', type=int,
        help='Number of hmm states per phone')
    parser.add_argument('outdir', help='Output directory')
    args = parser.parse_args()

    dict_phone = read_phonelist(args.phonefile)
    text_npz = np.load(args.phoneids)
    nstate_per_phone = args.nstate_per_phone
    outfile = args.outdir + '/states.int.npz'

    tot_states = list(range(len(dict_phone) * nstate_per_phone))
    dict_state_phone = dict((s, tot_states[s*nstate_per_phone:(s+1)*nstate_per_phone])
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
