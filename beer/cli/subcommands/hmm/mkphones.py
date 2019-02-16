'create a set of left-to-right HMM representing "phones"'

import argparse
from collections import defaultdict
import pickle

import torch
import yaml

import beer


def parse_topology(topology):
    state_ids = set()
    arcs = set()
    for arc_def in topology:
        start, end, weight = arc_def['start_id'], arc_def['end_id'], \
                             arc_def['trans_prob']
        state_ids.add(start), state_ids.add(end)
        arcs.add((start, end, weight))
    return sorted(list(state_ids)), arcs


def create_unit_graph(topology, start_pdf_id):
    state_ids, arcs = parse_topology(topology)
    non_emitting_states = (state_ids[0], state_ids[-1])
    graph = beer.graph.Graph()
    count = 0
    for state_id in range(len(state_ids)):
        if state_id in non_emitting_states:
            pdf_id = None
        else:
            pdf_id = start_pdf_id + count
            count += 1
        graph.add_state(pdf_id=pdf_id)
    graph.start_state = state_ids[0]
    graph.end_state = state_ids[-1]
    for arc in arcs: graph.add_arc(*arc)
    return graph, start_pdf_id + len(state_ids) - len(non_emitting_states)


def count_emitting_state(graph):
    count = 0
    for state_id in graph.states():
        if graph._states[state_id].pdf_id is not None: count += 1
    return count


def create_pdfs(mean, var, tot_emitting_states, group_conf):
    modelset = beer.NormalSet.create(
        mean=mean,
        cov=var,
        size=tot_emitting_states * group_conf['n_normal_per_state'],
        prior_strength=group_conf['prior_strength'],
        noise_std=group_conf['noise_std'],
        cov_type=group_conf['cov_type'],
        shared_cov=group_conf['shared_cov']
    )
    pdfs = beer.MixtureSet.create(tot_emitting_states, modelset,
                                  prior_strength=group_conf['prior_strength'])
    return pdfs


def setup(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--dataset', help='dataset for initialization')
    group.add_argument('-D', '--dimension', type=int,
                        help='dimension of features for the pdf')
    parser.add_argument('conf', help='configuration file')
    parser.add_argument('units', help='list of units to build and their corresponding group')
    parser.add_argument('out', help='output phone HMMs')


def main(args, logger):
    logger.debug(f'reading configuration file: {args.conf}')
    with open(args.conf, 'r') as f:
        conf = yaml.load(f)
    groups_conf = {group_conf['group_name']: group_conf for group_conf in conf}

    logger.debug(f'load the acoustic units name and group')
    with open(args.units, 'r') as f:
        grouped_unitnames = defaultdict(list)
        for line in f:
            name, group = line.strip().split()
            grouped_unitnames[group].append(name)

    if not args.dataset:
        logger.debug('no dataset provided assuming zero mean and ' \
                     'identity covariance matrix')
        mean, var = torch.zeros(args.dimension).float(), \
                    torch.ones(args.dimension).float()
    else:
        logger.debug(f'using "{args.dataset}" dataset for ' \
                     'initialization')
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
        mean, var = dataset.mean, dataset.var

    start_pdf_id = 0
    pdfs = []
    units = {}
    for group in grouped_unitnames:
        logger.debug(f'creating HMM for group "{group}"')
        tot_emitting_states = 0
        for name in grouped_unitnames[group]:
            logger.debug(f'creating HMM for unit "{name}"')
            group_conf = groups_conf[group]
            graph, start_pdf_id = create_unit_graph(group_conf['topology'],
                                                    start_pdf_id)
            units[name] = graph
            tot_emitting_states += count_emitting_state(graph)
        pdfs.append(create_pdfs(mean, var, tot_emitting_states, group_conf))
    emissions = beer.JointModelSet(pdfs)

    logger.debug('saving the HMMs on disk...')
    with open(args.out, 'wb') as f:
        pickle.dump((units, emissions), f)

    logger.info(f'created {len(units)} HMMs for a total of {len(emissions)}' \
                f' emitting states')
    logger.info(f'expected features dimension: {len(mean)}')


if __name__ == "__main__":
    main()

