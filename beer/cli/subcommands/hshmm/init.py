'initialize the phone loop depending on the HGSM'

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
    parser.add_argument('--adapt-from', help='language from the source languages with which to '
                        'initialize the target language subspace')
    parser.add_argument('--init-from', help='language from the source languages with which to '
                        'initialize the target language pdfvecs')
    parser.add_argument('--latent-nsamples', type=int, default=1,
                        help='number of embedding samples to use for initialization')
    parser.add_argument('--params-nsamples', type=int, default=1,
                        help='number of parameter samples to use for initialization')
    parser.add_argument('--same', action='store_true',
                        help='initialize all units using the same embedding')
    parser.add_argument('lang', help='name of language for which to initialize the phone-loop, used as a dictionary key')
    parser.add_argument('temp_gsm', help='temporary target language hgsm')
    parser.add_argument('gsm', help='multilingual hgsm to set the prior')
    parser.add_argument('posts', help='input latent posteriors')
    parser.add_argument('sploop', help='input subspace phone-loop')
    parser.add_argument('out_sploop', help='output subspace phone-loop')
    parser.add_argument('out_gsm', help='output hgsm with target language posteriors')


def main(args, logger):
    logger.debug('loading the GSM')
    lang = args.lang
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)
    
    with open(args.temp_gsm, 'rb') as f:
        target_gsm = pickle.load(f)
    if args.adapt_from:
        target_gsm['langs'][args.lang] = gsm['langs'][args.adapt_from]
    if args.init_from:
        _gsm = gsm['langs'][args.init_from]
    else:
        _gsm = target_gsm['langs'][lang]
    gsm['langs'] = target_gsm['langs']

    root_gsm = gsm['root']
    gsms_dict = gsm['langs']

    logger.debug('loading the units posterior')
    with open(args.posts, 'rb') as f:
        posts_dict, units_dict, nstates, groupidx = pickle.load(f)

    logger.debug('loading the subspace phoneloop')
    with open(args.sploop, 'rb') as f:
        sploop = pickle.load(f)

    logger.debug('loading the units')
    nunits = len(units_dict[lang])
    units_emissions = sploop[lang].modelset.original_modelset.modelsets[groupidx]
    if len(units_emissions)// nstates < 2:
        units_emissions = sploop[lang].modelset.original_modelset.modelsets[1 - groupidx]
    units = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    logger.info(f'initializing the phone loop')

    _gsm.transform.root_transform = root_gsm.shared_transform
    pdfvecs = _gsm.expected_pdfvecs(posts_dict[lang],
                                    latent_nsamples=args.latent_nsamples,
                                    params_nsamples=args.params_nsamples)
    if args.same:
        pdfvecs = [pdfvecs[0] for _ in pdfvecs]
    _gsm.update_models(units, pdfvecs)

        
    logger.debug('saving the subspace phoneloop')
    with open(args.out_sploop, 'wb') as f:
        pickle.dump(sploop, f)

    for lang, ploop in sploop.items():
        with open(args.out_sploop + '_' + lang, 'wb') as f:
            pickle.dump(ploop, f)

    with open(args.out_gsm, 'wb') as f:
        pickle.dump(gsm, f)


if __name__ == "__main__":
    main()

