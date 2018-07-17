
'''Convert phoneme transcription into integer sequences
'''

import argparse
import numpy as np
import os
from funcs import create_phone_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phonefile', type=str)
    parser.add_argument('transcription', type=str)
    parser.add_argument('nstate_per_phone', type=int)
    args = parser.parse_args()

    phonefile = args.phonefile
    trans_file = args.transcription
    nstate_per_phone = args.nstate_per_phone


    dict_phone_ids, dict_state_to_phone = create_phone_dict(phonefile,
                                                            nstate_per_phone)
    tot_states = list(range(len(dict_state_to_phone)))
    dict_map = dict((s, tot_states[s*nstate_per_phone:(s+1)*nstate_per_phone]) 
                    for s in range(len(dict_phone_ids)))
    outdir = os.path.dirname(os.path.abspath(trans_file))

    tmpdir = outdir + '/tmp/'
    try:
        os.stat(tmpdir)
    except:
        os.mkdir(tmpdir)

    with open(trans_file, 'r') as f:
        for l in f:
            l = l.strip()
            line = l.split()
            uttid = line.pop(0)
            utt = np.asarray([dict_map[dict_phone_ids[s]] for s in line]).reshape(-1)
           # dict_npz = dict(dict_npz, **{uttid:utt})
            np.save(tmpdir + uttid + '.npy', utt)


if __name__ == "__main__":
    main()
