'train a subspace phone-loop model'

import argparse
import copy
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
    parser.add_argument('-e', '--epochs', default=1, type=int,
                       help='number of training epochs (default: 1)')
    parser.add_argument('-c', '--learning-rate-cjg', default=1., type=float,
                        help='learning rate for the conjugate parameters '\
                             '(default: 1.)')
    parser.add_argument('-s', '--learning-rate-std', default=1e-1, type=float,
                        help='learning rate for the standard parameters '\
                             '(default: 1e-1)')
    parser.add_argument('-o', '--optim-state', help='optimizer state')
    parser.add_argument('-n', '--latent-nsamples', type=int, default=1,
                        help='number of samples for the latent posterior ' \
                             '(default: 1)')
    parser.add_argument('-k', '--params-nsamples', type=int, default=1,
                        help='number of samples for the parameters posterior ' \
                             '(default: 1)')
    parser.add_argument('--gpu', action='store_true', help='use a GPU')
    parser.add_argument('-r', '--logging_rate', type=int, default=100,
                        help='logging rate of the ELBO (default: 100)')
    parser.add_argument('gsm', help='input gsm')
    parser.add_argument('posts', help='input latent posteriors')
    parser.add_argument('sploop', help='input subspace phone-loop')
    parser.add_argument('out_gsm', help='output gsm')
    parser.add_argument('out_posts', help='output latent posteriors')
    parser.add_argument('out_sploop', help='output subspace phone-loop')


def main(args, logger):
    logger.debug('loading the GSM')
    with open(args.gsm, 'rb') as f:
        gsm = pickle.load(f)

    logger.debug('loading the units posterior')
    with open(args.posts, 'rb') as f:
        latent_posts, nunits, nstates, groupidx = pickle.load(f)

    logger.debug('loading the subspace phoneloop')
    with open(args.sploop, 'rb') as f:
        sploop = pickle.load(f)

    # Move to gpu here.
    # model = model.cuda()

    logger.debug('loading the units')
    units_emissions = sploop.modelset.original_modelset.modelsets[groupidx]
    units = [unit for unit in iterate_units(units_emissions, nunits, nstates)]

    logger.debug('building the optimizer')
    params = gsm.conjugate_bayesian_parameters(keepgroups=True)
    cjg_optim = beer.VBConjugateOptimizer(params, lrate=args.learning_rate_cjg)
    params = list(latent_posts.parameters()) + list(gsm.parameters())
    std_optim = torch.optim.Adam(params, lr=args.learning_rate_std)
    optim = beer.VBOptimizer(cjg_optim, std_optim)
    if args.optim_state and os.path.isfile(args.optim_state):
        logger.debug(f'loading optimizer state from: {args.optim_state}')
        state = torch.load(args.optim_state)
        optim.load_state_dict(state)

    for epoch in range(1, args.epochs + 1):
        optim.init_step()
        elbo = beer.evidence_lower_bound(gsm, units, latent_posts=latent_posts,
                                         latent_nsamples=args.latent_nsamples,
                                         params_nsamples=args.params_nsamples)
        elbo.backward()
        optim.step()

        if epoch % args.logging_rate == 0:
            logger.info(f'epoch={epoch:<20} elbo={float(elbo):<20}')

    logger.info(f'finished training at epoch={epoch} with elbo={float(elbo)}')

    # Move the model to cpu
    # model = model.cpu()

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

