'train a hierarchical subspace phone-loop model'

import argparse
import copy
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
    parser.add_argument('-c', '--learning-rate-cjg', default=1., type=float,
                        help='learning rate for the conjugate parameters '\
                             '(default: 1.)')
    parser.add_argument('-e', '--epochs', default=1, type=int,
                       help='number of training epochs (default: 1)')
    parser.add_argument('--gpu', action='store_true', help='use a GPU')
    parser.add_argument('-k', '--params-nsamples', type=int, default=1,
                        help='number of samples for the parameters posterior ' \
                             '(default: 1)')
    parser.add_argument('-n', '--lang-latent-nsamples', type=int, default=1,
                        help='number of samples for the language latent posteriors ' \
                             '(default: 1)')
    parser.add_argument('-u', '--unit-latent-nsamples', type=int, default=1,
                        help='number of samples for the unit latent posteriors ' \
                        '(default: 1)')
    parser.add_argument('--clip-grad', type=float, default=-1,
                        help='if set to a positive value, gradients will be clipped '
                        'to have a norm equal to this value')
    parser.add_argument('-o', '--optim-state', help='optimizer state')
    parser.add_argument('-p', '--skip-root-subspace', action='store_true',
                        help='freeze the parameters of the root subspace')
    parser.add_argument('-a', '--skip-language-posterior', action='store_true',
                        help='freeze the parameters of the language latent posteriors')
    parser.add_argument('-b', '--skip-unit-posterior', action='store_true',
                        help='freeze the parameters of the unit latent posteriors')
    parser.add_argument('-r', '--logging_rate', type=int, default=100,
                        help='logging rate of the ELBO (default: 100)')
    parser.add_argument('--use-sgd', '--sgd', action='store_true',
                        help='use SGD for optimization instead of Adam')
    parser.add_argument('-s', '--learning-rate-std', default=1e-1, type=float,
                        help='learning rate for the standard parameters '\
                             '(default: 1e-1)')
    parser.add_argument('gsm', help='input gsm')
    parser.add_argument('posts', help='input latent posteriors')
    parser.add_argument('sploop_to_lang', help='input subspace phone-loop')
    parser.add_argument('out_gsm', help='output gsm')
    parser.add_argument('out_posts', help='output latent posteriors')
    parser.add_argument('out_sploop', help='output subspace phone-loop')


def main(args, logger):
    if args.gpu:
        gpu_idx = beer.utils.reserve_gpu(logger=logger)

    logger.debug('loading the GSM')
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)
    root_gsm = gsm['root']
    gsms_dict = gsm['langs']
    for _gsm in gsms_dict.values():
        _gsm.transform.root_transform = root_gsm.shared_transform

    logger.debug('loading the units posterior')
    with open(args.posts, 'rb') as f:
        lp_data = pickle.load(f)
        posts_dict, units_dict, nstates, groupidx = lp_data
        labels = None


    logger.debug('loading the subspace phoneloop')
    phoneloops_dict = {}
    for line in open(args.sploop_to_lang):
        _sploop, lang = line.strip().split()
        with open(_sploop, 'rb') as f:
            phoneloops_dict[lang] = pickle.load(f)

    if args.gpu:
        logger.info(f'using gpu device: {gpu_idx}')
        for sploop in phoneloops_dict.values():
            sploop = sploop.cuda()
        root_gsm = root_gsm.cuda()
        for lang in gsms_dict.keys():
            gsms_dict[lang] = gsms_dict[lang].cuda()
            posts_dict[lang] = posts_dict[lang].cuda()

    logger.debug('loading the units')
    units_emissions_dict = {}

    for lang, sploop in phoneloops_dict.items():
        units_emissions_dict[lang] = sploop.modelset.original_modelset.modelsets[groupidx]
        for group in sploop.modelset.original_modelset.modelsets:
            if len(group) > len(units_emissions_dict[lang]):
                units_emissions_dict[lang] = group
        units_dict[lang] = [unit for unit in iterate_units(units_emissions_dict[lang], len(units_dict[lang]), nstates)]

    logger.debug('building the optimizer')
    if args.skip_root_subspace:
        params = [[]]
    else:
        params = sum([list(_gsm.conjugate_bayesian_parameters(keepgroups=True)) for _gsm in gsms_dict.values()], [])
    cjg_optim = beer.VBConjugateOptimizer(params, lrate=args.learning_rate_cjg)
    

    params = []
    params = list(root_gsm.parameters())

    for lang, _gsm in gsms_dict.items():
        params += list(_gsm.transform.latent_posterior.parameters())
    all_latent_posts = []

    for lang, latent_posts in posts_dict.items():
        all_latent_posts.append(latent_posts)
        params += list(latent_posts.parameters())
    if not args.use_sgd:
        std_optim = torch.optim.Adam(params, lr=args.learning_rate_std)
    else:
        std_optim = torch.optim.SGD(params, lr=args.learning_rate_std)
    
    optim = beer.VBOptimizer(cjg_optim, std_optim)
    if args.optim_state and os.path.isfile(args.optim_state):
        logger.debug(f'loading optimizer state from: {args.optim_state}')
        if args.gpu:
            maplocation = 'cuda'
        else:
            maplocation = 'cpu'
        state = torch.load(args.optim_state, maplocation)
        optim.load_state_dict(state)

    # listify all the dictionaries for training
    models_and_submodels = []
    for lang, _gsm in gsms_dict.items():
        lang_units = [unit for i, unit in enumerate(units_dict[lang])]
        logger.debug(f'{lang}: {len(lang_units)}')
        models_and_submodels.append([_gsm, lang_units])
    kwargs = {
        'univ_latent_nsamples': args.lang_latent_nsamples,
        'latent_posts': all_latent_posts,
        'latent_nsamples': args.unit_latent_nsamples,
        'params_nsamples': args.params_nsamples,
    }

    for epoch in range(1, args.epochs + 1):
        optim.init_step()
        elbo = beer.evidence_lower_bound(root_gsm, models_and_submodels, **kwargs)
        elbo.backward()
        if args.skip_root_subspace:
            root_gsm.zero_grad()
        if args.skip_language_posterior:
            for lang, _gsm in gsms_dict.items():
                _gsm.transform.latent_posterior.zero_grad()
        if args.skip_unit_posterior:
            for lang, latent_posts in posts_dict.items():
                latent_posts.zero_grad()
        if args.clip_grad > 0:
            for grp in optim.std_optim.param_groups:
                norm = torch.nn.utils.clip_grad_norm_(grp['params'], args.clip_grad)
        else:
            norm = 0
        optim.step()

        if args.logging_rate > 0 and epoch % args.logging_rate == 0:
            logger.info(f'epoch={epoch:<20} elbo={float(elbo):<20}')
            logger.debug(f'epoch={epoch:<20} norm={float(norm):<20}')

    logger.info(f'finished training at epoch={epoch} with elbo={float(elbo)}')

    if args.gpu:
        for sploop in phoneloops_dict.values():
            sploop = sploop.cpu()
        root_gsm = root_gsm.cpu()
        for lang in gsms_dict.keys():
            gsms_dict[lang] = gsms_dict[lang].cpu()
            posts_dict[lang] = posts_dict[lang].cpu()

    logger.debug('saving the HGSM')
    gsm['root'] = root_gsm
    gsm['langs'] = gsms_dict
    with open(args.out_gsm, 'wb') as f:
        pickle.dump(gsm, f)

    logger.debug('saving the units posterior')
    with open(args.out_posts, 'wb') as f:
        pickle.dump((posts_dict, units_dict, nstates, groupidx), f)

    logger.debug('saving the subspace phoneloop')
    sploop = phoneloops_dict
    with open(args.out_sploop, 'wb') as f:
        pickle.dump(sploop, f)
    for lang, sploop in phoneloops_dict.items():
        with open(args.out_sploop + '_' + lang, 'wb') as f:
            pickle.dump(sploop, f)

    if args.optim_state:
        logger.debug(f'saving the optimizer state to: {args.optim_state}')
        torch.save(optim.state_dict(), args.optim_state)


if __name__ == "__main__":
    main()

