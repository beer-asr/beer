
'''Create a neural network from a YAML configuration file. The
components of the network are build upon pytorch "modules" object.
See in the "conf" directory for an example.
'''


import argparse
import logging
import pickle
import yaml
import beer


logging.basicConfig(format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', default='',
                        help='comma separated list of key=value to '
                             'replace in the configuration file')
    parser.add_argument('conf', help='YAML configuration file of the '
                                     'features')
    parser.add_argument('out', help='output file')
    args = parser.parse_args()

    # Load the formatting values.
    format_values = {}
    for key_val in args.set.strip().split(','):
        try:
            key, val = key_val.split('=')
            format_values[key.strip()] = val.strip()
        except Exception:
            pass

    # Load the configuration.
    with open(args.conf, 'r') as fid:
        conf_str = fid.read().format(**format_values)
        conf = yaml.load(conf_str)

    # Create the neural network.
    nnet = beer.nnet.create(conf)

    # Save the model on disk.
    with open(args.out, 'wb') as fid:
        pickle.dump(nnet, fid)


if __name__ == '__main__':
    main()

