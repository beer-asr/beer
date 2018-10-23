'Create a the HMM graph and the corresponding emssions.'

import numpy as np
import argparse
import beer
import pickle
import torch
import yaml
import logging

logging.basicConfig(format='%(levelname)s: %(message)s')


def parse_topology(topology):
    arcs = []
    for arc_conf in topology:
        start = arc_conf['start_id']
        end = arc_conf['end_id']
        weight = arc_conf['trans_prob']
        arcs.append((start, end, weight))
    return arcs


def create_unit_graph(n_states, arcs, start_pdf_id):
    graph = beer.graph.Graph()
    states = []
    current_pdf_id = start_pdf_id
    for i in range(n_states):
        if i == 0 or i == n_states - 1:
            pdf_id = None
        else:
            pdf_id = current_pdf_id
            current_pdf_id += 1
        states.append(graph.add_state(pdf_id=pdf_id))
    graph.start_state = states[0]
    graph.end_state = states[-1]

    for start, end, weight in arcs:
        graph.add_arc(start, end, weight)

    graph.normalize()
    return graph, start_pdf_id + n_states - 2

def create_gmm_emissions(group, mean, var):
    tot_states = group['n_units'] * (group['n_state_per_unit'] - 2)
    if group['shared_emissions']:
        modelset = beer.NormalSet.create(
            mean, var,
            size=group['n_normal_per_state'],
            prior_strength=group['prior_strength'],
            noise_std=group['noise_std'],
            cov_type=group['cov_type'],
            shared_cov=group['shared_cov']
        )
        modelset = beer.RepeatedModelSet(modelset, tot_states)
    else:
        modelset = beer.NormalSet.create(
            mean, var,
            size=group['n_normal_per_state'] * tot_states,
            prior_strength=group['prior_strength'],
            noise_std=group['noise_std'],
            cov_type=group['cov_type'],
            shared_cov=group['shared_cov']
        )
    return modelset


def create_lds_emissions(group, mean, var):
    tot_states = group['n_units'] * (group['n_state_per_unit'] - 2)
    if group['shared_emissions']:
        modelset = beer.LDSSet.create(
            mean,
            size=1,
            prior_strength=group['prior_strength'],
            noise_std=group['noise_std'],
            variance=group['variance'],
            n_dct_bases=group['n_dct_bases'],
            memory=group['memory']
        )
        modelset = beer.RepeatedModelSet(modelset, tot_states)
    else:
        modelset = beer.LDSSet.create(
            mean,
            size=tot_states,
            prior_strength=group['prior_strength'],
            noise_std=group['noise_std'],
            variance=group['variance'],
            n_dct_bases=group['n_dct_bases'],
            memory=group['memory']
        )
    return modelset


def create_emissions(group, mean, var):
    emission_type = group['emission_type']

    if emission_type == 'GMM':
        modelset = create_gmm_emissions(group, mean, var)
    elif emission_type == 'LDS':
        modelset = create_lds_emissions(group, mean, var)
    else:
        logging.error(f'Unknown emission_type {emission_type}')
        exit(1)
    return modelset


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stats', help='Feature statistics file for hmm model')
    group.add_argument('--dim', type=int,
                        help='Dimension of feature, used for vae-hmm model')
    parser.add_argument('conf', help='Configuration file')
    parser.add_argument('phones', help='list of phones')
    parser.add_argument('hmm_graphs', help='hmm graph for each unit')
    parser.add_argument('emissions', help='outout emissions')
    args = parser.parse_args()

    # Load the HMM configuration.
    with open(args.conf, 'r') as fid:
        conf = yaml.load(fid)

    # Load the phones.
    phones = []
    with open(args.phones, 'r') as fid:
        for line in fid:
            tokens = line.split()
            phones.append(tokens[0])
    phones = phones

    # Get the data statistics.
    if args.stats:
        stats = np.load(args.stats)
        mean = torch.from_numpy(stats['mean']).float()
        var = torch.from_numpy(stats['var']).float()
    else:
        dim = args.dim
        mean = torch.zeros(dim).float()
        var = torch.ones(dim).float()

    unit_count = 0
    pdf_id = 0
    units = {}
    emissions = []
    for group in conf:

        # Create the unit graphs.
        for i in range(group['n_units']):
            unit_id = unit_count
            unit_count += 1
            nstates = group['n_state_per_unit']
            arcs = parse_topology(group['topology'])
            unit, pdf_id = create_unit_graph(nstates, arcs, pdf_id)
            units[phones[unit_id]] = unit

        tot_states = group['n_units'] * (group['n_state_per_unit'] - 2)
        modelset = create_emissions(group, mean, var)
        modelset = beer.MixtureSet.create(tot_states, modelset)
        emissions.append(modelset)

    # Merge the pdfs into a single set.
    emissions = beer.JointModelSet(emissions)

    with open(args.hmm_graphs, 'wb') as fid:
        pickle.dump(units, fid)

    with open(args.emissions, 'wb') as fid:
        pickle.dump(emissions, fid)

if __name__ == '__main__':
    main()

