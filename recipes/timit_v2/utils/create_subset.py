

import random
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Random choose N \
        utterances to create subset')
    parser.add_argument('srcdir', help='Source directory')
    parser.add_argument('n_utt', type=int, help='Number of utternces in subset')
    parser.add_argument('tgtdir', help='Target directory')
    args = parser.parse_args()

    srcdir = args.srcdir
    n_utt = args.n_utt
    tgtdir = args.tgtdir

    src_feats_file = os.path.join(srcdir, 'feats.npz')
    src_trans = os.path.join(srcdir, 'trans')
    src_phone_int = os.path.join(srcdir, 'phones.int.npz')
    tgt_feats_file = os.path.join(tgtdir, 'feats.npz')
    tgt_trans = os.path.join(tgtdir, 'trans')
    tgt_phone_int = os.path.join(tgtdir, 'phones.int.npz')
    src_feats = np.load(src_feats_file)
    src_phones = np.load(src_phone_int)
    src_keys = list(src_feats.keys())
    random.shuffle(src_keys)

    if not os.path.exists(tgtdir):
        os.makedirs(tgtdir)
    with open(src_trans, 'r') as f:
        seqs = [l.rstrip('\n') for l in f]
    dict_src_trans = {s.split()[0]: s.split()[1:] for s in seqs}

    dict_tgt_feat = {}
    dict_tgt_phones = {}
    with open(tgt_trans, 'w') as f:
        for i in range(n_utt):
            utt = src_keys[i]
            ft = src_feats[utt]
            phone_int = src_phones[utt]
            trans = ' '.join(dict_src_trans[utt])
            print(utt, trans, file=f)
            dict_tgt_feat[utt] = ft
            dict_tgt_phones[utt] = phone_int
    np.savez(tgt_feats_file, **dict_tgt_feat)
    np.savez(tgt_phone_int, **dict_tgt_phones)

if __name__ == "__main__":
    main()



