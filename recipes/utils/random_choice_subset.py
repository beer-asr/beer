

import random
import numpy as np
import sys
import os

datadir = sys.argv[1]
num_utts = int(sys.argv[2])
tgtdir = sys.argv[3]

feats_file = datadir + '/feats.npz'
labels_file = datadir + '/states.int.npz'
trans = datadir + '/phones.text'

os.makedirs(tgtdir, exist_ok=False)

feats = np.load(feats_file)
labels = np.load(labels_file)

feats = np.load(feats_file)
labels = np.load(labels_file)

keys = list(feats.keys())
random.shuffle(keys)

tmpftdir = tgtdir + '/tmp_ft/'
tmplabdir = tgtdir + '/tmp_lab/'
os.makedirs(tmpftdir, exist_ok=False)
os.makedirs(tmplabdir, exist_ok=False)

with open(trans, 'r') as f:
    seqs = [l.rstrip('\n') for l in f]
dict_trans = {s.split()[0]: s.split()[1:] for s in seqs}

with open(tgtdir + '/phones.text', 'w') as f:
    for i in range(num_utts):
        utt = keys[i]
        ft = feats[utt]
        lab = labels[utt]
        tran = ' '.join(dict_trans[utt])
        f.write(utt + ' ' + tran + '\n')
        np.save(tmpftdir + utt + '.npy', ft)
        np.save(tmplabdir + utt + '.npy', lab)





