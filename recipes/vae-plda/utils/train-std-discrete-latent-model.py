
'''Generic training for standard Bayesian models with non-sequential
discrete latent variable (i.e. Mixture model).  The script assumes the
database is stored in "numpy" archives with the data in the "features"
key and (optional) labels in the "labels" key.

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
            batch_size=batch_size, shuffle=True)
        for mb_data, mb_labels in dataloader:
            yield mb_data, mb_labels


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='size of the mini-batch. If set to a '
                             'negative value, the whole database '
                             'will be used for each update')
    parser.add_argument('--use-gpu', action='store_true',
                        help='use GPU for training the model')
    parser.add_argument('--lrate', type=float, default=1.,
                        help='learning rate')
    parser.add_argument('--logging-rate', type=int, default=10,
                        help='number of messages between log messages')
    parser.add_argument('--dir-tmp-models', default=None,
                        help='directory where to store the '
                             'intermediate models. If provided '
                             'a model will be stored every '
                             '"logging_rate"')
    parser.add_argument('dbstats', help='statistics of the database')
    parser.add_argument('fealist', help='list of npz archives')
    parser.add_argument('model', help='model to train')
    parser.add_argument('output', help='output path for the trained model')
    args = parser.parse_args()

    # Load the list of features.
    with open(args.fealist, 'r') as fid:
        archives = [line.strip() for line in fid]

    # Load the statistics of the databse.
    db_stats = np.load(args.dbstats)

    # Load the model to train.
    with open(args.model, 'rb') as fid:
        model = pickle.load(fid)

    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    logging_msg = 'epoch: {} batch: {} ln p(X) >= {:.3f}'

    # Build the optimizer.
    params = model.grouped_parameters
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(*params,
                                                            lrate=args.lrate)

    if args.batch_size > 0:
        bsize = args.batch_size
    else:
        bsize = int(db_stats['counts'])
    for epoch in range(args.epochs):
        for batch_no, data in enumerate(batches(archives, bsize,
                                                to_torch_dataset)):
            features, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            elbo = beer.evidence_lower_bound(model, features,
                                             datasize=float(db_stats['counts']),
                                             labels=labels)
            elbo.natural_backward()
            optimizer.step()
            if (batch_no + 1) % args.logging_rate == 0:
                print(logging_msg.format(epoch + 1, batch_no + 1,
                                         float(elbo) / float(db_stats['counts'])))
                # If the user has provided a directory, we store the
                # intermediate models.
                if args.dir_tmp_models:
                    model_name = 'epoch{}_batch{}.mdl'.format(epoch + 1,
                                                              batch_no + 1)
                    path = os.path.join(args.dir_tmp_models, model_name)

                    # Move the model on CPU before and store it.
                    tmp_model = model.to(torch.device('cpu'))
                    with open(path, 'wb') as fid:
                        pickle.dump(tmp_model, fid)

    # Move the model on CPU before and store it.
    model = model.to(torch.device('cpu'))
    with open(args.output, 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    run()

