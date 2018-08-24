
'''Create emission models
'''

import numpy as np
import argparse
import beer
import pickle
import torch
import yaml
import logging

logging.basicConfig(format='%(levelname)s: %(message)s')

mdl_conf = {
    'n_units': 48,
    'n_state_per_unit': 3,
    'n_normal_per_state': 8,
    'prior_strength': 1,
    'noise_std': 0.1,
    'cov_type': 'diagonal',
    'shared_cov': False
    }



def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stats', type=str,
        help='Feature statistics file for hmm model')
    group.add_argument('--dim', type=int,
        help='Dimension of feature, used for vae-hmm model')
    parser.add_argument('emis_conf', type=str, help='Configuration file')
    parser.add_argument('model', type=str)
    args = parser.parse_args()

    with open(args.emis_conf, 'r') as conf:
        newconf = yaml.load(conf)

    # Check for unknown options
    for key in newconf:
        if key not in mdl_conf:
            logging.error('Unknown setting "{}"'.format(key))
            exit(1)
    mdl_conf.update(newconf)

    if args.stats:
        stats = np.load(args.stats)
        mean = torch.from_numpy(stats['mean']).float()
        var = torch.from_numpy(stats['var']).float()
    else:
        dim = args.dim
        mean = torch.zeros(dim).float()
        var = torch.ones(dim).float()
    mdl = args.model

    tot_states = mdl_conf['n_units'] * mdl_conf['n_state_per_unit']
    normal_size = tot_states * mdl_conf['n_normal_per_state']

    normalset = beer.NormalSet.create(mean, var, size=normal_size,
                                      prior_strength=mdl_conf['prior_strength'],
                                      noise_std=mdl_conf['noise_std'],
                                      cov_type=mdl_conf['cov_type'],
                                      shared_cov=mdl_conf['shared_cov'])
    mixtureset = beer.MixtureSet.create(tot_states, normalset)
    with open(mdl, 'wb') as m:
        pickle.dump(mixtureset, m)

if __name__ == '__main__':
    main()
