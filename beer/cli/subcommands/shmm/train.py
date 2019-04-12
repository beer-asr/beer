'train a subspace phone-loop model'

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
    parser.add_argument('-n', '--latent-nsamples', type=int, default=1,
                        help='number of samples for the latent posterior ' \
                             '(default: 1)')
    parser.add_argument('-o', '--optim-state', help='optimizer state')
    parser.add_argument('-p', '--posteriors', action='store_true',
                        help='train the latent posteriors only')
    parser.add_argument('-r', '--logging_rate', type=int, default=1000,
                        help='logging rate of the ELBO (default: 1000)')
    parser.add_argument('-s', '--learning-rate-std', default=1e-1, type=float,
                        help='learning rate for the standard parameters '\
                             '(default: 1e-1)')
    parser.add_argument('gsm', help='input gsm')
    parser.add_argument('posts', help='input latent posteriors')
    parser.add_argument('sploop', help='input subspace phone-loop')
    parser.add_argument('out_gsm', help='output gsm')
    parser.add_argument('out_posts', help='output latent posteriors')
    parser.add_argument('out_sploop', help='output subspace phone-loop')


def main(args, logger):
    if args.gpu:
        gpu_idx = beer.utils.reserve_gpu(logger=logger)

    logger.debug('loading the GSM')
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)

    logger.debug('loading the units posterior')
    with open(args.posts, 'rb') as f:
        lp_data = pickle.load(f)
        if len(lp_data) == 4:
            latent_posts, nunits, nstates, groupidx = lp_data
            labels = None
        else:
            logger.debug('using labels for training the latent prior')
            latent_posts, nunits, nstates, groupidx, labels = lp_data


    logger.debug('loading the subspace phoneloop')
    with open(args.sploop, 'rb') as f:
        sploop = pickle.load(f)

    if args.gpu:
        logger.info(f'using gpu device: {gpu_idx}')
        sploop = sploop.cuda()
        gsm = gsm.cuda()
        latent_posts = latent_posts.cuda()
        if labels is not None:
            labels = labels.cuda()

    logger.debug('loading the units')
    units_emissions = sploop.modelset.original_modelset.modelsets[groupidx]
    units = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    logger.debug('building the optimizer')
    if args.posteriors:
        params = [[]]
    else:
        params = gsm.conjugate_bayesian_parameters(keepgroups=True)
    cjg_optim = beer.VBConjugateOptimizer(params, lrate=args.learning_rate_cjg)
    if args.posteriors:
        params = list(latent_posts.parameters())
    else:
        params = list(latent_posts.parameters()) + list(gsm.parameters())
    std_optim = torch.optim.Adam(params, lr=args.learning_rate_std)
    optim = beer.VBOptimizer(cjg_optim, std_optim)
    if args.optim_state and os.path.isfile(args.optim_state):
        logger.debug(f'loading optimizer state from: {args.optim_state}')
        if args.gpu:
            maplocation = 'cuda'
        else:
            maplocation = 'cpu'
        state = torch.load(args.optim_state, maplocation)
        optim.load_state_dict(state)

    kwargs = {
        'latent_posts': latent_posts,
        'latent_nsamples': args.latent_nsamples,
        'params_nsamples': args.params_nsamples,
    }
    if labels is not None:
        kwargs['labels'] = labels
    for epoch in range(1, args.epochs + 1):
        optim.init_step()
        elbo = beer.evidence_lower_bound(gsm, units, **kwargs)
        elbo.backward()
        optim.step()

        if args.logging_rate > 0 and epoch % args.logging_rate == 0:
            logger.info(f'epoch={epoch:<20} elbo={float(elbo):<20}')

    logger.info(f'finished training at epoch={epoch} with elbo={float(elbo)}')

    if args.gpu:
        sploop = sploop.cpu()
        gsm = gsm.cpu()
        latent_posts = latent_posts.cpu()
        if labels is not None:
            labels = labels.cpu()

    logger.debug('saving the GSM')
    with open(args.out_gsm, 'wb') as f:
        pickle.dump(gsm, f)

    logger.debug('saving the units posterior')
    with open(args.out_posts, 'wb') as f:
        if labels is None:
            pickle.dump((latent_posts, nunits, nstates, groupidx), f)
        else:
            pickle.dump((latent_posts, nunits, nstates, groupidx, labels), f)

    logger.debug('saving the subspace phoneloop')
    with open(args.out_sploop, 'wb') as f:
        pickle.dump(sploop, f)

    if args.optim_state:
        logger.debug(f'saving the optimizer state to: {args.optim_state}')
        torch.save(optim.state_dict(), args.optim_state)


if __name__ == "__main__":
    main()

