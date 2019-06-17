'set the prior of the GSM from the posterior of another GSM'

import argparse
import copy
import math
import os
import pickle
import sys

import torch

import beer



def setup(parser):
    parser.add_argument('-v', '--var', default=1, type=float,
                        help='variance of the new prior (default: 1)')
    parser.add_argument('gsm', help='gsm of which to set the prior')
    parser.add_argument('gsm_post', help='gsm to get the posterior from')
    parser.add_argument('out_gsm', help='output gsm')


def main(args, logger):
    logger.debug('loading the GSM')
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)

    logger.debug('loading the GSM (post)')
    with open(args.gsm_post, 'rb') as f:
        gsm_post = pickle.load(f)

    logger.debug('setting the new prior')
    logvar = math.log(args.var)
    gsm.transform.weights.prior.params.mean = \
        gsm_post.transform.weights.posterior.params.mean.data
    log_diag_cov = gsm.transform.weights.posterior.params.log_diag_cov.data
    gsm.transform.weights.prior.params.log_diag_cov = \
        torch.zeros_like(log_diag_cov) + logvar
    gsm.transform.bias.prior.params.mean = \
        gsm_post.transform.bias.posterior.params.mean.data
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

    logger.debug('saving the GSM')
    with open(args.out_gsm, 'wb') as f:
        pickle.dump(gsm, f)


if __name__ == "__main__":
    main()

