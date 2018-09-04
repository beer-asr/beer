'Create a VAE-HMM model.'

import argparse
import logging
import pickle
import yaml
import numpy as np
import torch
import beer


logging.basicConfig(format='%(levelname)s: %(message)s')


encoder_normal_layer = {
    'isotropic': beer.nnet.NormalIsotropicCovarianceLayer,
    'diagonal': beer.nnet.NormalDiagonalCovarianceLayer
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-cov-type',
                        choices=['isotropic', 'diagonal'],
                        help='type of covariance for the encoder.')
    parser.add_argument('--decoder-cov-type',
                        choices=list(encoder_normal_layer.keys()),
                        help='type of covariance for the decoder.')
    parser.add_argument('stats', help='training data stats for ' \
                                      'initialization')
    parser.add_argument('encoder_out_dim', type=int,
                        help='dimension of output of the encoder.')
    parser.add_argument('latent_dim', type=int,
                        help='dimension of the latent space')
    parser.add_argument('encoder', help='encoder network')
    parser.add_argument('nflow', help='normalizing flow network')
    parser.add_argument('latent_model', help='model over the latent space')
    parser.add_argument('decoder', help='decoder network')
    parser.add_argument('out', help='output model')
    args = parser.parse_args()

    stats = np.load(args.stats)

    with open(args.encoder, 'rb') as fid:
        encoder = pickle.load(fid)

    with open(args.nflow, 'rb') as fid:
        nnet_flow, flow_params_dim = pickle.load(fid)

    with open(args.latent_model, 'rb') as fid:
        latent_model = pickle.load(fid)

    with open(args.decoder, 'rb') as fid:
        decoder = pickle.load(fid)

    prob_layer = encoder_normal_layer[args.encoder_cov_type]
    enc_prob_layer = prob_layer(args.encoder_out_dim, args.latent_dim)

    nflow = beer.nnet.InverseAutoRegressiveFlow(
        dim_in=args.encoder_out_dim,
        flow_params_dim=flow_params_dim,
        normal_layer=enc_prob_layer,
        nnet_flow=nnet_flow
    )

    data_mean = torch.from_numpy(stats['mean']).float()
    data_var = torch.from_numpy(stats['var']).float()
    normal = beer.Normal.create(data_mean, data_var,
                                cov_type=args.decoder_cov_type)

    vae = beer.VAEGlobalMeanVariance(
        encoder,
        nflow,
        decoder,
        normal,
        latent_model
    )

    with open(args.out, 'wb') as fid:
        pickle.dump(vae, fid)


if __name__ == '__main__':
    main()

