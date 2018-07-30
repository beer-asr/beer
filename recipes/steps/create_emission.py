
'''Create emission models
'''

import numpy as np
import argparse
import sys
sys.path.insert(0, '../../beer')
import beer
import pickle
import torch
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('emis_conf', type=str, help='Configuration file')
    parser.add_argument('stats', type=str, help='data statistics')
    parser.add_argument('model_dir', type=str)
    args = parser.parse_args()

    with open(args.emis_conf, 'r') as conf:
        conf = yaml.load(conf)
    stats = np.load(args.stats)
    mdl = args.model_dir + '/emission.mdl'

    mean = torch.from_numpy(stats['mean']).float()
    var = torch.from_numpy(stats['var']).float()
    modelset = beer.create_model(conf, mean, var)

    with open(mdl, 'wb') as m:
        pickle.dump(modelset, m)

if __name__ == '__main__':
    main()
