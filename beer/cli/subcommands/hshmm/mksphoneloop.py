
'create a hierarchical subspace phone-loop model'

import argparse
import copy
import pickle
import sys

import torch
import yaml

import beer


# Create a view of the emissions (aka modelset) for each units.
def iterate_units(modelset, nunits, nstates):
    for idx in range(nunits):
        start, end = idx * nstates, (idx + 1) * nstates
        yield modelset[start:end]


# Compute the sufficient statistics from the natural parameters.
def param_stats(param):
    post, prior = param.posterior, param.prior
    return post.natural_parameters() - prior.natural_parameters()


# Init the statistics of the weights parameters to be uniformly
# distributed.
def init_weights_stats(weights):
    stats = weights.stats
    counts = stats[:, -1].sum()
    ncomps = stats.shape[-1]
    new_stats = torch.zeros_like(stats)
    new_stats[:] = counts / ( 3 * ncomps)
    new_stats[:, -1] = counts / 3
    return new_stats


# Init the statistics of the mean/precision parameters to be the
# identical for each component of a HMM's state.
def init_means_precisions_stats(means_precisions, weights):
    pi = weights.value().view(-1, 1)
    stats = means_precisions.stats
    stats.shape, pi / pi.sum()
    new_stats = torch.zeros_like(stats)
    new_stats[:] = (stats * pi).sum(dim=0, keepdim=True)
    return new_stats


def setup(parser):
    parser.add_argument('-c', '--classes',
                        help='assign a broad class to each unit')
    parser.add_argument('-g', '--unit-group', default='speech-unit',
                       help='group of unit to model with the subspace ' \
                            '(default:"speech-unit")')
    parser.add_argument('-p', '--posteriors',
                        help='posteriors to use for initialization')
    parser.add_argument('-l', '--latent-dim', default=2, type=int,
                        help='dimension of the language latent space (default:2)')
    parser.add_argument('-t', '--unit-latent-dim', default=2, type=int,
                        help='dimension of the language specific unit latent spaces'
                        '(default:2)')
    parser.add_argument('-d', '--dlatent-dim', default=2, type=int,
                        help='dimension of the discriminant latent space (default:2)')
    parser.add_argument('conf', help='configuration file use to create '
                                     'the phone-loop')
    parser.add_argument('phoneloop_to_lang',
                        help='a space separated file containing mappings from phoneloop to languages')
    parser.add_argument('gsm', help='output generalized subspace model')
    parser.add_argument('posts', help='output units posterior')
    parser.add_argument('sploop', help='output subspace Phone-Loop')


def main(args, logger):
    logger.debug(f'reading configuration file: {args.conf}')
    with open(args.conf, 'r') as f:
        conf = yaml.load(f)

    logger.debug(f'loading the configuration for the group: {args.unit_group}')
    for groupidx, groupconf in enumerate(conf):
        if groupconf['group_name'] == args.unit_group:
            break
    else:
        raise ValueError(f'No matching group "{args.unit_group}"')
    groupidx = 0
    logger.debug(f'group configuration index {groupidx}')

    ncomps = int(groupconf['n_normal_per_state'])
    logger.debug(f'number of Gauss. components per state: {ncomps}')

    topology = groupconf['topology']
    nstates = 0
    for arc in topology:
        nstates = max(nstates, arc['end_id'] - 1)
        nstates
    logger.debug(f'number of states per unit: {nstates}')

    logger.debug('loading the phone-loop')
    lang_to_phoneloop = {line.strip().split()[1]: line.strip().split()[0]
                         for line in open(args.phoneloop_to_lang)}
    phoneloops_dict = {}
    units_emissions_dict = {}
    nunits_dict = {}
    for lang, _phoneloop in lang_to_phoneloop.items():
        with open(_phoneloop, 'rb') as f:
            phoneloops_dict[lang] = pickle.load(f)
        units_emissions_dict[lang] = phoneloops_dict[lang].modelset.original_modelset.modelsets[groupidx]
        for group in phoneloops_dict[lang].modelset.original_modelset.modelsets:
            if len(group) > len(units_emissions_dict[lang]):
                units_emissions_dict[lang] = group
        nunits_dict[lang] = len(units_emissions_dict[lang]) // nstates

    ## logger.debug('loading the units models')

    nunits = ','.join([f'{lang}: {n}' for lang, n in nunits_dict.items()])
    logger.debug(f'number of units to include in the subspace: {nunits}')

    # These steps are needed as in the training of the standard HMM
    # there is no guarantee that the statistics will be retained by
    # the parameters.
    logger.debug('initializing the parameters\' sufficient statistics')
    for lang, units_emissions in units_emissions_dict.items():
        for param in units_emissions.bayesian_parameters():
            param.stats = param_stats(param)

    labels = None
    nclasses = 1


    logger.debug('create the latent normal prior')
    latent_prior = beer.Normal.create(torch.zeros(args.latent_dim),
                                      torch.ones(args.latent_dim),
                                      cov_type = 'full')
    unit_prior = beer.Normal.create(torch.zeros(args.unit_latent_dim),
                                    torch.ones(args.unit_latent_dim), cov_type='full')

    logger.debug('setting the parameters to be handled by the subspace')

    units_dict = {}
    for lang, units_emissions in units_emissions_dict.items():
        nunits = nunits_dict[lang]
        newparams = {
            param: beer.SubspaceBayesianParameter.from_parameter(param, unit_prior)
            for param in units_emissions.bayesian_parameters()
        }
        units_emissions.replace_parameters(newparams)

        for unit in iterate_units(units_emissions, nunits, nstates):
            weights = unit.categoricalset.weights
            means_precisions = unit.modelset.means_precisions
            weights.stats = init_weights_stats(weights)
            means_precisions.stats = init_means_precisions_stats(means_precisions,
                                                                 weights)

        units_dict[lang] = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    all_langs = [lang for lang in lang_to_phoneloop.keys()]

    # Dictionaries mapping each language to its GSM and unit posteriors
    posts_dict = {}
    gsms_dict = {}

    logger.debug('creating the GSMs')
    tpl = copy.deepcopy(units_dict[all_langs[0]][0])

    # Create root GSM (hyper-subspace)
    _gsm = beer.GSM.create(tpl, args.unit_latent_dim, unit_prior)
    _transform = _gsm.transform
    univ_affine_transform = beer.AffineTransform.create(args.latent_dim,
                                                        _transform.out_dim * (_transform.in_dim + 1))
    root_gsm = beer.HierarchicalGSM(univ_affine_transform, latent_prior)
    # Define language specific GSMs
    for lang, phoneloop in lang_to_phoneloop.items():
        pseudo_transform = beer.HierarchicalAffineTransform.create(latent_prior, args.unit_latent_dim,
                                                                   _transform.out_dim,
                                                                   univ_affine_transform,
                                                                   cov_type='diagonal')
        lang_units = units_dict[lang]
        gsms_dict[lang] = beer.GSM.create(tpl, args.unit_latent_dim, unit_prior)
        gsms_dict[lang].transform = pseudo_transform
        posts_dict[lang] = gsms_dict[lang].new_latent_posteriors(len(lang_units))
        pdfvecs = gsms_dict[lang].expected_pdfvecs(posts_dict[lang])
        gsms_dict[lang].update_models(lang_units, pdfvecs)

    gsm = {'root': root_gsm,
           'langs': gsms_dict,
           'lang_latent_dim': args.latent_dim,
           'unit_latent_dim': args.unit_latent_dim,
           'params_dim': _transform.out_dim,
    }
    logger.debug('saving the GSM')
    with open(args.gsm, 'wb') as f:
        pickle.dump(gsm, f)

    logger.debug('saving the units posterior')
    with open(args.posts, 'wb') as f:
        pickle.dump((posts_dict, units_dict, nstates, groupidx), f)

    logger.debug('saving the subspace phoneloop')
    ploop = phoneloops_dict
    with open(args.sploop, 'wb') as f:
        pickle.dump(ploop, f)

    for lang, ploop in phoneloops_dict.items():
        with open(args.sploop + '_' + lang, 'wb') as f:
            pickle.dump(ploop, f)

    lens = [len(x) for x in units_dict.values()]
    logger.info(f'created {len(posts_dict)} GSMs (latent dim: {args.latent_dim})')
    logger.info('num_units:')
    for lang, x in units_dict.items():
        logger.info(f'\t {lang}: {len(x)}')
    logger.info(f'latent prior: {latent_prior}')
    logger.info(f'unit latent prior: {unit_prior}')


if __name__ == "__main__":
    main()

