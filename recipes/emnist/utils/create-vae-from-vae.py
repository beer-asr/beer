
'''Create a VAE model from a configuration file and another VAE.'''


import argparse
import pickle
import yaml
import numpy as np
import torch
import beer


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('conf', help='condfiguration file')
    parser.add_argument('dbstats', help='statistics of the training database')
    parser.add_argument('init_model',
                        help='model to initialize the encoder/decoder')
    parser.add_argument('output', help='output file for the model')
    args = parser.parse_args()

    # Load the statistics of the training data.
    dbstats = np.load(args.dbstats)
    mean, variance = dbstats['mean'], dbstats['variance']
    mean = torch.from_numpy(mean).float()
    variance = torch.from_numpy(variance).float()

    # Load the model configuration file.
    with open(args.conf, 'r') as fid:
        data_str = fid.read().replace('<feadim>', str(len(mean)))
        conf = yaml.load(data_str)

    # Create the model.
    model = beer.create_model(conf, mean, variance)

    # Replace the encoder / decoder from the given VAE
    # model.
    with open(args.init_model, 'rb') as fid:
        init_model = pickle.load(fid)
    model.encoder = init_model.encoder
    model.nflow
    model.decoder = init_model.decoder

    # Save the model.
    with open(args.output, 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    run()

