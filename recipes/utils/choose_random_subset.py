

import random
import numpy as np
import sys
import os

datadir = sys.argv[1]
num_utts = int(sys.argv[2])
tgtdir = sys.argv[3]

feats_file = datadir + '/feats.npz'
labels_file = datadir + '/phones.int.npz'
stats_file = datadir + '/feats_stats.npz'

os.makedirs(tgtdir, exist_ok=True)

feats = np.load(feats_file)
labels = np.load(labels_file)

feats = np.load(feats_file)
labels = np.load(labels_file)

keys = list(feats.keys())
random.shuffle(keys)

tmpftdir = tgtdir + '/tmp_ft/'
tmplabdir = tgtdir + '/tmp_lab/'
os.makedirs(tmpftdir, exist_ok=True)
os.makedirs(tmplabdir, exist_ok=True)

for i in range(num_utts):
    utt = keys[i]
    ft = feats[utt]
    lab = labels[utt]
    np.save(tmpftdir + utt + '.npy', ft)
    np.save(tmplabdir + utt + '.npy', lab)

