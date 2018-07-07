
'''Classification accuracy for standard Bayesian models with
non-sequential discrete latent variable (i.e. Mixture model) embedded
in a Variational Auto Encoder.
'''

import argparse
import copy
import os
import random
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import beer



def to_torch_dataset(np_features, np_labels):
    fea, labels = torch.from_numpy(np_features).float(), \
        torch.from_numpy(np_labels).long()
    return torch.utils.data.TensorDataset(fea, labels)

def batches(archives_list, batch_size, to_torch_dataset):
    arlist = copy.deepcopy(archives_list)
    random.shuffle(arlist)
    for archive in arlist:
        data = np.load(archive)
        dataset = to_torch_dataset(data['features'], data['labels'])
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size)
        for mb_data, mb_labels in dataloader:
            yield mb_data, mb_labels


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='size of the mini-batch. If set to a '
                             'negative value, the whole database '
                             'will be used for each update')
    parser.add_argument('--nsamples', type=int, default=1,
                        help='number of samples to estimate the '
                             'expectation')
    parser.add_argument('--use-gpu', action='store_true', help='use GPU')
    parser.add_argument('fealist', help='list of npz archives')
    parser.add_argument('model', help='model to train')
    parser.add_argument('outfile', help='output file')
    args = parser.parse_args()

    # Load the list of features.
    with open(args.fealist, 'r') as fid:
        archives = [line.strip() for line in fid]

    # Load the model to train.
    with open(args.model, 'rb') as fid:
        model = pickle.load(fid)

    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # Batch size for the stochastic training.
    if args.batch_size > 0:
        bsize = args.batch_size
    else:
        bsize = int(db_stats['counts'])

    hits = 0
    counts = 0
    for batch_no, data in enumerate(batches(archives, bsize,
                                            to_torch_dataset)):
        features, labels = data[0].to(device), data[1].to(device)
        print(features.shape, labels.shape)
        means, variances = model.encoder(features)
        samples = beer.sample_from_normals(means, variances, args.nsamples)
        samples = samples.view(args.nsamples * len(features), -1)
        posts = model.latent_model.posteriors(samples)
        posts = posts.view(args.nsamples, len(features), -1).mean(dim=0)
        predictions = posts.argmax(dim=1)
        hits += int((predictions == labels).sum())
        counts += len(labels)

    with open(args.outfile, 'w') as fid:
        print('# frames:', int(counts), file=fid)
        print('accuracy:', round(100 * float(hits) / float(counts), 3),
              file=fid)

if __name__ == '__main__':
    run()

