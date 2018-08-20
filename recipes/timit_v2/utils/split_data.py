

import random
import numpy as np
import os
import argparse

def split_data(data, n_job):
    keys = list(data.keys())
    tot_keys = len(keys)
    key_chunks = np.array_split(np.array(range(tot_keys)), n_job)
    for k in key_chunks:
        yield dict((keys[i], data[keys[i]]) for i in k)


def main():
    parser = argparse.ArgumentParser(description='Split data into N subsets')
    parser.add_argument('srcdir', help='Source directory')
    parser.add_argument('num_job', type=int, help='Number of subsets')
    args = parser.parse_args()

    srcdir = args.srcdir
    num_job = args.num_job
    src_feats_file = os.path.join(srcdir, 'feats.npz')
    src_trans = os.path.join(srcdir, 'trans')
    src_phone_file = os.path.join(srcdir, 'phones.int.npz')
    src_feats = np.load(src_feats_file)
    src_phones = np.load(src_phone_file)
    with open(src_trans, 'r') as f:
        seqs = [l.rstrip('\n') for l in f]
    dict_src_trans = {s.split()[0]: ' '.join(s.split()[1:]) for s in seqs}
    tgtdir = os.path.join(srcdir, 'split' + str(num_job))

    # Split data
    split_feats = split_data(src_feats, num_job)
    split_phones = split_data(src_phones, num_job)
    split_trans = split_data(dict_src_trans, num_job)

    for i, (ft, ph, tr) in enumerate(zip(split_feats, split_phones, split_trans)):
        sub_dir = os.path.join(tgtdir, str(i+1))
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        sub_feats = os.path.join(sub_dir, 'feats.npz')
        sub_phones = os.path.join(sub_dir, 'phones.int.npz')
        sub_trans = os.path.join(sub_dir, 'trans')
        np.savez(sub_feats, **ft)
        np.savez(sub_phones, **ph)
        with open(sub_trans, 'w') as f:
            for utt in tr.keys():
                print(utt, tr[utt], file=f)

if __name__ == "__main__":
    main()



