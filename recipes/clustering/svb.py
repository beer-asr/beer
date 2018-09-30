'Stochastic Variational Bayes training.'

import argparse
import glob
import os
import pickle
import random
import gc

import numpy as np
import torch

import beer

def compute_kl_div(model, data_dir, max_batches=10):
    path = os.path.join(data_dir, 'batch*.npz')
    kl_mean = 0.
    counts = 0
    with torch.no_grad():
        for i, path in enumerate(glob.glob(path)):
            if i >= max_batches:
                break
            X = torch.from_numpy(np.load(path)['features']).float()
            if X.shape[0] == 0:
                continue
            encoder_states = model.encoder(X)
            posterior_params = model.encoder_problayer(encoder_states)
            samples, post_llh = model.encoder_problayer.samples_and_llh(
                posterior_params, use_mean=True)

            # Per-frame KL divergence between the (approximate) posterior
            # and the prior.
            latent_stats = model.latent_model.sufficient_statistics(samples)
            prior_llh = model.latent_model.expected_log_likelihood(latent_stats)

            kl_divs = post_llh - prior_llh
            kl_mean += kl_divs.sum()
            counts += len(X)
            del X
            gc.collect()
    return kl_mean / counts

def log_pred(model, data_dir, max_batches=50):
    path = os.path.join(data_dir, 'batch*.npz')
    retval = 0.
    counts = 0
    with torch.no_grad():
        for i, path in enumerate(glob.glob(path)):
            if i >= max_batches:
                break
            X = torch.from_numpy(np.load(path)['features']).float()
            if X.shape[0] == 0:
                continue
            elbo = beer.collapsed_evidence_lower_bound(model, X)
            retval += float(elbo)
            counts += len(X)
            del X, elbo
    return retval / counts


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain-epochs', type=int, default=1,
                        help='number of pre-training epochs')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--lrate', type=float, default=1e-1,
                        help='learning rate for the nnet parameters')
    parser.add_argument('--nnet-lrate', type=float, default=1e-3,
                        help='learning rate for the nnet parameters')
    parser.add_argument('--update-prior', action='store_true',
                        help='update the prior')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='weigth decay for the nnet parameters')
    parser.add_argument('init_model', help='initial model')
    parser.add_argument('train_data_dir', help='training data directory')
    parser.add_argument('test_data_dir', help='test data directory')
    parser.add_argument('out_model', help='output model')
    args = parser.parse_args()

    # Retrieve the list of batches.
    batches = list(glob.glob(os.path.join(args.train_data_dir, 'batch*npz')))

    # Compute the total number of points of the data base
    tot_counts = 0
    for batch in batches:
        tot_counts += np.load(batch)['features'].shape[0]

    # Load the model.
    with open(args.init_model, 'rb') as f:
        model = pickle.load(f)

    # Prepare the optimizer.
    std_optimizer = torch.optim.Adam(
        model.modules_parameters(),
        lr=args.nnet_lrate,
        weight_decay=args.weight_decay
    )

    if args.update_prior:
        params = model.mean_field_factorization()
    else:
        params = model.normal.mean_field_factorization()
    optimizer = beer.BayesianModelOptimizer(
        params,
        lrate=args.lrate,
        std_optim=std_optimizer
    )

    # To monitor the convergence.
    elbos = []
    klds = []
    log_preds = []
    def callback(model, epoch, elbo_value):
        kld = compute_kl_div(model, args.test_data_dir)
        elbos.append(elbo_value)
        klds.append(kld)
        l_pred = log_pred(model, args.test_data_dir)
        log_preds.append(l_pred)
        print(f'epoch={epoch}/{args.epochs} ln p(X) >= {elbo_value:.2f}  ' \
            f'D(q || p) = {kld:.2f} (nats)  ' \
            f'ln p(X_test|X_train) = {l_pred:.2f}')

    epoch = 0
    while epoch < args.epochs:
        # Randomized the order of the batches.
        batch_list = list(batches)
        random.shuffle(batch_list)
        elbo_value = 0.
        if epoch + 1 <= args.pretrain_epochs:
            kwargs = {'kl_weight': 0.}
        else:
            kwargs = {'kl_weight': 1.}
        for batch in batch_list:
            X = torch.from_numpy(np.load(batch)['features']).float()
            optimizer.init_step()
            elbo = beer.evidence_lower_bound(model, X, datasize=tot_counts,
                                             **kwargs)
            elbo.backward()
            optimizer.step()
            elbo_value += float(elbo) / tot_counts
            del X, elbo
            gc.collect()

        epoch += 1

        if callback is not None:
            callback(model, epoch, elbo_value / len(batches))

    with open(args.out_model, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    run()
