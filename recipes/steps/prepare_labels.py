
'''Convert phoneme transcription into integer sequences
'''

import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phonelist', type=str)
    parser.add_argument('transcription', type=str)
    parser.add_argument('nstate_per_phone', type=int)
    args = parser.parse_args()

    map_file = args.phonelist
    trans_file = args.transcription
    nstate_per_phone = args.nstate_per_phone


    dict_phone = {}
    with open(map_file, 'r') as f:
        for l in f:
            l = l.strip()
            phones = l.split()
            phone = phones.pop(0)
            dict_phone[phone] = phones[0]
    
    dict_npz = {}
    
    outdir = os.path.dirname(os.path.abspath(trans_file))
    try:
        os.stat(outdir + '/tmp')
    except:
        os.mkdir(outdir + '/tmp')
    
    with open(trans_file, 'r') as f:
        for l in f:
            l = l.strip()
            line = l.split()
            uttid = line.pop(0)
            utt = np.asarray([[int(dict_phone[i])] * nstate_per_phone \
                for i in line]).reshape(-1)
           # dict_npz = dict(dict_npz, **{uttid:utt})
            tmpdir = outdir + '/tmp/'
            np.save(tmpdir + uttid + '.npy', utt)


if __name__ == "__main__":
    main()
