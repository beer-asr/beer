'Train a SMM model.'

import argparse
import pickle
import random

import numpy as np
import torch

import beer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--latent-dim', type=int, default=2,
                        help='dimension of the latent space')
    parser.add_argument('fea', help='features')
    parser.add_argument('model', help='output model')
    args = parser.parse_args()

    fea = np.load(args.fea)
    X = torch.from_numpy(fea['fea']).float()
    y = torch.from_numpy(fea['labels']).long()

    # Latent prior (GMM).
    modelset = beer.NormalSet.create(
        mean=torch.zeros(args.latent_dim),
        cov=torch.ones(args.latent_dim),
        size=int(y.max()) + 1,
        cov_type='full'
    )
    latent_prior = beer.Mixture.create(modelset)

    # Multinomial model.
    mean = torch.ones(X.shape[1]) / X.shape[1]
    model = beer.Categorical.create(mean)
    newparams = {
        param: beer.SubspaceBayesianParameter.from_parameter(param, latent_prior)
        for param in model.bayesian_parameters()
    }
    model.replace_parameters(newparams)

    # Generalized Subspace Model.
    gsm = beer.GSM.create(model, args.latent_dim, latent_prior)

    # Model instances.
    models, latent_posts = gsm.new_models(len(X), cov_type='diagonal')

    # Accumulate the statistics.
    for i, model in enumerate(models):
        dim = X.shape[1]
        data = torch.eye(dim)
        data[range(dim), range(dim)] = X[i]
        elbo = beer.evidence_lower_bound(model, data)
        elbo.backward(std_params=False)

    params = gsm.conjugate_bayesian_parameters(keepgroups=True)
    cjg_optim = beer.VBConjugateOptimizer(params, lrate=1e-1)
    params = list(latent_posts.parameters()) + list(gsm.parameters())
    std_optim = torch.optim.Adam(params, lr=1e-1)
    optim = beer.VBOptimizer(cjg_optim, std_optim)

    epochs = 500
    batch_size = 100
    for epoch in range(epochs):
        optim.init_step()
        elbo = beer.evidence_lower_bound(datasize=len(models))
        for batch in range(0, len(X), batch_size):
            #batch_idxs = random.sample(range(len(X)), k=batch_size)
            #batch_models = [models[s] for s in batch_idxs]
            #batch_latent_posts = latent_posts[batch_idxs]
            #batch_labels = y[batch_idxs]

            batch_models = models[batch:batch+batch_size]
            batch_latent_posts = latent_posts[batch:batch+batch_size]
            batch_labels = y[batch:batch+batch_size]

            optim.init_step()
            elbo += beer.evidence_lower_bound(
                gsm,
                batch_models,
                labels=batch_labels,
                latent_posts=batch_latent_posts,
                latent_nsamples=5,
                params_nsamples=5,
                datasize=len(models)
            )
        elbo.backward()
        optim.step()

        means = latent_posts.params.mean
        pred = latent_prior.posteriors(means).argmax(dim=1).detach().numpy()
        print(f'Accuracy: {100 * np.mean(pred == y.numpy()):.5f} %')



if __name__ == '__main__':
    main()

