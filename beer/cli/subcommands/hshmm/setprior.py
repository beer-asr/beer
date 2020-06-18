'set the prior of the GSM from the posterior of another GSM'

import argparse
import copy
import math
import os
import pickle
import sys

import torch

import beer


# Create a view of the emissions (aka modelset) for each units.
def iterate_units(modelset, nunits, nstates):
    for idx in range(nunits):
        start, end = idx * nstates, (idx + 1) * nstates
        yield modelset[start:end]


def setup(parser):
    parser.add_argument('-v', '--var', default=1, type=float,
                        help='variance of the new prior (default: 1)')
    parser.add_argument('gsm', help='gsm of which to set the prior')
    parser.add_argument('posts', help='input latent posteriors')
    parser.add_argument('sploop', help='input subspace phone-loop')
    parser.add_argument('gsm_init', help='gsm to get the posterior from')
    parser.add_argument('posts_init', help='posteriosr for initialization')
    parser.add_argument('out_gsm', help='output gsm')
    parser.add_argument('out_posts', help='output latent posteriors')
    parser.add_argument('out_sploop', help='output subspace phone-loop')


def main(args, logger):
    logger.debug('loading the GSM')
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)

    logger.debug('loading the subspace phoneloop')
    with open(args.sploop, 'rb') as f:
        sploop = pickle.load(f)

    logger.debug('loading the GSM (init)')
    with open(args.gsm_init, 'rb') as f:
        gsm_init = pickle.load(f)

    logger.debug('loading the units posterior')
    with open(args.posts, 'rb') as f:
        latent_posts, nunits, nstates, groupidx = pickle.load(f)

    logger.debug('loading the units posterior (init)')
    with open(args.posts_init, 'rb') as f:
        latent_posts_init, _, _, _ = pickle.load(f)

    logger.debug('loading the units')
    units_emissions = sploop.modelset.original_modelset.modelsets[groupidx]
    units = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    logger.info(f'initializing the GSM prior with variance {args.var}')

    logger.debug('setting the new prior')
    logvar = math.log(args.var)
    gsm.transform.weights.prior.params.mean = \
        gsm_init.transform.weights.posterior.params.mean.data
    log_diag_cov = gsm.transform.weights.posterior.params.log_diag_cov.data
    gsm.transform.weights.prior.params.log_diag_cov = \
        torch.zeros_like(log_diag_cov) + logvar
    gsm.transform.bias.prior.params.mean = \
        gsm_init.transform.bias.posterior.params.mean.data
    log_diag_cov = gsm.transform.bias.posterior.params.log_diag_cov.data
    gsm.transform.bias.prior.params.log_diag_cov = \
        torch.zeros_like(log_diag_cov) + logvar

    logger.debug('initializing the posterior')
    w_params = gsm.transform.weights.prior.params
    b_params = gsm.transform.bias.prior.params
    gsm.transform.weights.posterior.params.mean = \
        torch.nn.Parameter(w_params.mean.clone())
    gsm.transform.weights.posterior.params.log_diag_cov = \
        torch.nn.Parameter(w_params.log_diag_cov.clone())
    gsm.transform.bias.posterior.params.mean = \
        torch.nn.Parameter(b_params.mean.clone())
    gsm.transform.bias.posterior.params.log_diag_cov = \
        torch.nn.Parameter(b_params.log_diag_cov.clone())


    logger.info(f'initializing the unit posteriors')
    min_d = min(len(latent_posts), len(latent_posts_init))
    latent_posts.params.mean.data[:min_d, :] = \
        latent_posts_init.params.mean.data[:min_d, :]
    latent_posts.params.log_diag_cov.data[:min_d, :] = \
        latent_posts_init.params.log_diag_cov.data[:min_d, :]
    if min_d < len(latent_posts):
        diag_cov = gsm_init.latent_prior.cov.diag()
        mean = gsm_init.latent_prior.mean
        in_dim = gsm.transform.in_dim
        noise = torch.randn(len(latent_posts) - min_d, in_dim)
        new_means = mean[None, :] + noise * diag_cov.sqrt()[None, :]
        latent_posts.params.mean.data[min_d:, :] = new_means

    pdfvecs = gsm.expected_pdfvecs(latent_posts)
    gsm.update_models(units, pdfvecs)


    logger.info(f'initializing the unit prior')
    gsm.latent_prior = gsm_init.latent_prior

    logger.debug('saving the GSM')
    with open(args.out_gsm, 'wb') as f:
        pickle.dump(gsm, f)

    logger.debug('saving the units posterior')
    with open(args.out_posts, 'wb') as f:
        pickle.dump((latent_posts, nunits, nstates, groupidx), f)

    logger.debug('saving the subspace phoneloop')
    with open(args.out_sploop, 'wb') as f:
        pickle.dump(sploop, f)


if __name__ == "__main__":
    main()

