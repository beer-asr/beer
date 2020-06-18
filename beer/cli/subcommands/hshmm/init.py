'initialize the phone loop depending on the GSM'

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
    parser.add_argument('lang', help='name of language for which to initialize the phone-loop, used as a dictionary key')
    parser.add_argument('gsm', help='gsm of which to set the prior')
    parser.add_argument('posts', help='input latent posteriors')
    parser.add_argument('sploop', help='input subspace phone-loop')
    parser.add_argument('out_sploop', help='output subspace phone-loop')


def main(args, logger):
    logger.debug('loading the GSM')
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)
    root_gsm = gsm['root']
    gsms_dict = gsm['langs']

    lang = args.lang
    logger.debug('loading the units posterior')
    with open(args.posts, 'rb') as f:
        posts_dict, unit_id_to_lang, nunits, nstates, groupidx = pickle.load(f)

    logger.debug('loading the subspace phoneloop')
    with open(args.sploop, 'rb') as f:
        sploop = pickle.load(f)

    logger.debug('loading the units')
    units_emissions = sploop.modelset.original_modelset.modelsets[groupidx]
    units = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    logger.info(f'initializing the phone loop')
    # _gsm = beer.GSM.create(units[0], args.unit_latent_dim, unit_prior)
    # _transform = _gsm.transform
    # univ_affine_transform = beer.AffineTransform.create(args.latent_dim,
    #                                                     _transform.out_dim * (_transform.in_dim + 1))

    # pseudo_transform = beer.HierarchicalAffineTransform.create(root_gsm.latent_prior,
    #                                                            gsm['unit_latent_dim'],
    #                                                            gsm['params_dim'],
    #                                                            root_gsm.shared_transform,
    #                                                            cov_type='diagonal')
    # _gsm = beer.GSM.create(tpl, args.unit_latent_dim, unit_prior)
    # _gsm.transform = pseudo_transform
    # _lang_posts = _gsm.new_latent_posteriors(len(units))

    _gsm = gsms_dict['lang']
    pdfvecs = _gsm.expected_pdfvecs(latent_posts)
    _gsm.update_models(units, pdfvecs)

    logger.debug('saving the subspace phoneloop')
    with open(args.out_sploop, 'wb') as f:
        pickle.dump(sploop, f)


if __name__ == "__main__":
    main()

