
'Create a normalizing flow network from a YAML configuration file.'


import argparse
import logging
import pickle
import yaml
import torch
import beer


logging.basicConfig(format='%(levelname)s: %(message)s')

normal_layer = {
    'isotropic': beer.nnet.NormalIsotropicCovarianceLayer,
    'diagonal': beer.nnet.NormalDiagonalCovarianceLayer
}


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
            key, val = key_val.strip().split('=')
            format_values[key] = val
        except Exception:
            pass

    # Load the configuration.
    with open(args.conf, 'r') as fid:
        conf_str = fid.read().format(**format_values)
        conf = yaml.load(conf_str)

    activation = beer.nnet.create_nnet_element(conf['activation'])
    nnet_flow = []
    for i in range(conf['depth']):
        nnet_flow.append(
            beer.nnet.AutoRegressiveNetwork(
                dim_in=conf['dim'],
                flow_params_dim=conf['flow_params_dim'],
                depth=conf['block_depth'],
                width=conf['block_width'],
                activation=activation
            )
        )
    nnet_flow = torch.nn.Sequential(*nnet_flow)

    # Save the model on disk.
    with open(args.out, 'wb') as fid:
        pickle.dump((nnet_flow, conf['flow_params_dim']), fid)


if __name__ == '__main__':
    main()

