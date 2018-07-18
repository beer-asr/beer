'''Generic training for a Variational Auto-Encoder with
a Normal distribution as prior. The script assumes the database is
stored in "numpy" archives with the data in the "features" key.

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


LOG_MSG= 'epoch: {} batch: {} ln p(X) >= {:.3f}'


def to_torch_dataset(np_features):
    fea = torch.from_numpy(np_features).float()
    return torch.utils.data.TensorDataset(fea)

def batches(archives_list, batch_size, to_torch_dataset):
    arlist = copy.deepcopy(archives_list)
    random.shuffle(arlist)
    for archive in arlist:
        data = np.load(archive)
        dataset = to_torch_dataset(data['features'])
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=True)
        for mb_data in dataloader:
            yield mb_data[0]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_argument_group('global arguments')
    group.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    group.add_argument('--batch-size', type=int, default=-1,
                        help='size of the mini-batch. If set to a '
                             'negative value, the whole database '
                             'will be used for each update')
    group.add_argument('--logging-rate', type=int, default=10,
                        help='number of messages between log messages')
    group.add_argument('--dir-tmp-models', default=None,
                        help='directory where to store the '
                             'intermediate models. If provided '
                             'a model will be stored every '
                             '"logging_rate"')
    group.add_argument('--use-gpu', action='store_true',
                        help='use GPU for training the model')
    group = parser.add_argument_group('std. model specific arguments')
    group.add_argument('--lrate', type=float, default=1.,
                        help='learning rate')
    group = parser.add_argument_group('VAE specific arguments')
    group.add_argument('--lrate-nnet', type=float, default=1e-3,
                        help='learning rate for the encoder/decoder')
    group.add_argument('--weight-decay', type=float, default=1e-2,
                        help='L2 regularizer')
    group.add_argument('--nsamples', type=int, default=1,
                        help='number of sample of the '
                             'reparameterization trick')
    group.add_argument('--kl-weight', type=float, default=1.,
                        help='weight for the KL term')
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

    # Build the optimizer.
    nnet_parameters = list(model.encoder.parameters()) + \
        list(model.decoder.parameters())
    params = model.grouped_parameters
    std_optim = torch.optim.Adam(nnet_parameters, lr=args.lrate_nnet,
                                 weight_decay=args.weight_decay)
    optimizer = beer.BayesianModelCoordinateAscentOptimizer(*params,
                                                            lrate=args.lrate,
                                                            std_optim=std_optim)

    # Batch size for the stochastic training.
    if args.batch_size > 0:
        bsize = args.batch_size
    else:
        bsize = int(db_stats['counts'])

    for epoch in range(args.epochs):
        for batch_no, data in enumerate(batches(archives, bsize,
                                                to_torch_dataset)):

            features = data.to(device)
            optimizer.zero_grad()
            elbo = beer.evidence_lower_bound(model, features,
                                             datasize=float(db_stats['counts']),
                                             kl_weight=args.kl_weight,
                                             nsamples=args.nsamples)
            elbo.backward()
            elbo.natural_backward()
            optimizer.step()
            if (batch_no + 1) % args.logging_rate == 0:
                print(LOG_MSG.format(epoch + 1, batch_no + 1,
                                     float(elbo) / float(db_stats['counts'])))

                # If the user has provided a directory, we store the
                # intermediate models.
                if args.dir_tmp_models:
                    model_name = 'epoch{}_batch{}.mdl'.format(epoch + 1,
                                                              batch_no + 1)
                    path = os.path.join(args.dir_tmp_models, model_name)

                    # Move the model on CPU before and store it.
                    model = model.to(torch.device('cpu'))
                    with open(path, 'wb') as fid:
                        pickle.dump(model, fid)
                    model = model.to(torch.device(device))

    # Move the model on CPU before and store it.
    model = model.to(torch.device('cpu'))
    with open(args.output, 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    run()

