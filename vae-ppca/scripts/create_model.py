
"""Create a Bayesian PPCA model from a YAML configuration file."""

import argparse
import pickle
import ppca
import yaml


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='YAML configuration file')
    parser.add_argument('model', help='output model')
    args = parser.parse_args()

    # Load the configuration file.
    with open(args.config, 'r') as fid:
        config = yaml.load(fid)

    # Create the model.
    model = ppca.PPCA.create(**config)

    # Save it on disk.
    with open(args.model, 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    run()
