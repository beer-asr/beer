
'create a subspace phone-loop model'

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
    parser.add_argument('-l', '--latent-dim', default=2, type=int,
                        help='dimension of the latent space (default:2)')
    parser.add_argument('-d', '--dlatent-dim', default=2, type=int,
                        help='dimension of the discriminant latent space (default:2)')
    parser.add_argument('conf', help='configuration file use to create ' \
                                     'the phone-loop')
    parser.add_argument('phoneloop', help='phone-loop to initialize the ' \
                                          'subspace from')
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
    with open(args.phoneloop, 'rb') as f:
        ploop = pickle.load(f)

    logger.debug('loading the units models')
    units_emissions = ploop.modelset.original_modelset.modelsets[groupidx]

    nunits = len(units_emissions) // nstates
    logger.debug(f'number of units to include in the subspace: {nunits}')

    # This steps is needed as in the training of the standard HMM
    # there is no guarantee that the statistics will be retained by
    # the parameters.
    logger.debug('initializing the parameters\' sufficient statistics')
    for param in units_emissions.bayesian_parameters():
        param.stats = param_stats(param)

    labels = None
    nclasses = 1
    if args.classes:
        logger.debug(f'extracting the units\' class from: {args.classes}')
        with open(args.classes, 'r') as f:
            unit2class = {}
            class2unit = {}
            for line in f:
                unit, uclass = line.strip().split()
                unit2class[unit] = uclass
                class2unit[uclass] = unit
        classnames = [classname for classname in class2unit]
        class2idx = {uclass: i for i, uclass in enumerate(sorted(classnames))}
        nclasses = len(classnames)
        labels = torch.zeros(nunits).long()
        unit_idx = 0
        for unit in ploop.start_pdf:
            if unit in unit2class:
                labels[unit_idx] = class2idx[unit2class[unit]]
                unit_idx += 1


    logger.debug('create the latent normal prior')
    latent_prior = beer.Normal.create(torch.zeros(args.latent_dim),
                                      torch.ones(args.latent_dim),
                                      cov_type = 'full')

    logger.debug('setting the parameters to be handled by the subspace')
    newparams = {
        param: beer.SubspaceBayesianParameter.from_parameter(param, latent_prior)
        for param in units_emissions.bayesian_parameters()
    }
    units_emissions.replace_parameters(newparams)
    for unit in iterate_units(units_emissions, nunits, nstates):
        weights = unit.weights
        means_precisions = unit.modelset.means_precisions
        weights.stats = init_weights_stats(weights)
        means_precisions.stats = init_means_precisions_stats(means_precisions,
                                                             weights)

    logger.debug('creating the units model')
    units = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    logger.debug('creating the GSM')
    tpl = copy.deepcopy(units[0])
    if labels is None:
        gsm = beer.GSM.create(tpl, args.latent_dim, latent_prior)
        latent_posts = gsm.new_latent_posteriors(len(units))
        pdfvecs = gsm.expected_pdfvecs(latent_posts)
        gsm.update_models(units, pdfvecs)
    else:
        dlatent_prior = beer.Normal.create(torch.zeros(args.dlatent_dim),
                                           torch.ones(args.dlatent_dim),
                                           cov_type = 'full')

        gsmset = beer.GSMSet.create(tpl, nclasses, args.latent_dim,
                                    args.dlatent_dim, latent_prior,
                                    dlatent_prior)
        latent_posts = gsmset.new_latent_posteriors(len(units))
        pdfvecs = gsmset.expected_pdfvecs(latent_posts)
        gsmset.update_models(units, pdfvecs)
        gsm = beer.Mixture.create(gsmset)

    logger.debug('saving the GSM')
    with open(args.gsm, 'wb') as f:
        pickle.dump(gsm, f)

    logger.debug('saving the units posterior')
    with open(args.posts, 'wb') as f:
        if labels is None:
            pickle.dump((latent_posts, nunits, nstates, groupidx), f)
        else:
            pickle.dump((latent_posts, nunits, nstates, groupidx, labels), f)

    logger.debug('saving the subspace phoneloop')
    with open(args.sploop, 'wb') as f:
        pickle.dump(ploop, f)

    logger.info(f'created {nunits} subspace HMMs (latent dim: {args.latent_dim})')
    logger.info(f'latent prior: {latent_prior}')


if __name__ == "__main__":
    main()

