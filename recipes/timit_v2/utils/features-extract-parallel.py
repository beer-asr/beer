
'''Extract speech features from a list of wav files read from stdin.
The file will be provided as a "pipe" command if the line of the given
file end up with "|" (without quotes).

'''


import argparse
import beer
import io
import logging
import os
import subprocess
import sys

import yaml
import numpy as np
from scipy.io.wavfile import read



logging.basicConfig(format='%(levelname)s: %(message)s')


feaconf = {
    'srate': 16000,
    'preemph': 0.97,
    'window_len': 0.025,
    'framerate': 0.01,
    'apply_fbank': True,
    'nfilters': 26,
    'cutoff_hfreq': 8000,
    'cutoff_lfreq': 20,
    'apply_deltas': True,
    'delta_order': 2,
    'delta_winlen': 2,
    'apply_dct': True,
    'n_dct_coeff': 13,
    'lifter_coeff': 22,
    'utt_mnorm': True,
}


def compute_dct_bases(nfilters, n_dct_coeff):
    dct_bases = np.zeros((nfilters, n_dct_coeff))
    for m in range(n_dct_coeff):
        dct_bases[:, m] = np.cos((m+1) * np.pi / nfilters * (np.arange(nfilters) + 0.5))
    return dct_bases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feaconf', help='configuration file of the '
                                        'features')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    # Override the default configuration.
    with open(args.feaconf, 'r') as fid:
        new_conf = yaml.load(fid)

    # Check for unknown options.
    for key in new_conf:
        if key not in feaconf:
            logging.error('Unknown setting "{}"'.format(key))
            exit(1)
    feaconf.update(new_conf)

    # Pre-compute the DCT bases.
    dct_bases = compute_dct_bases(feaconf['nfilters'], feaconf['n_dct_coeff'])

    for line in sys.stdin:
        tokens = line.strip().split()
        uttid, inwav = tokens[0], ' '.join(tokens[1:])

        # If 'inwav' ends up with the '|' symbol, 'inwav' is
        # interpreted as a command otherwise we assume 'inwav' to
        # be a path to a wav file.
        if inwav[-1] == '|':
            proc = subprocess.run(inwav[:-1], shell=True,
                                  stdout=subprocess.PIPE)
            sr, signal = read(io.BytesIO(proc.stdout))
        else:
            sr, signal = read(inwav)
        if not sr == feaconf['srate']:
            msg = 'Sampling rate ({}) does not match the one ' \
                  'of the given file ({}).'
            logging.error(msg.format(feaconf['srate'], sr))
            exit(1)

        # Mel spectrum.
        features, fft_len = beer.features.short_term_mspec(
            signal,
            flen=feaconf['window_len'],
            frate=feaconf['framerate'],
            preemph=feaconf['preemph'],
            srate=feaconf['srate'],
        )

        # Filter bank.
        if feaconf['apply_fbank']:
            fbank = beer.features.create_fbank(feaconf['nfilters'], fft_len,
                                               lowfreq=feaconf['cutoff_lfreq'],
                                               highfreq=feaconf['cutoff_hfreq'])
            features = features @ fbank.T

        # Take the logarithm of the magnitude spectrum.
        features = np.log(1e-6 + features)

        # DCT transform.
        if feaconf['apply_dct']:
            features = features @ dct_bases

            # HTK compatibility steps (probably doesn't change
            # the accuracy of the recognition).
            features *= np.sqrt(2. / feaconf['nfilters'])

            # Liftering.
            l_coeff = feaconf['lifter_coeff']
            lifter = 1 + (l_coeff / 2) * np.sin(np.pi * \
                (1 + np.arange(feaconf['n_dct_coeff'])) / l_coeff)
            features *= lifter

        # Deltas.
        if feaconf['apply_deltas']:
            delta_order = feaconf['delta_order']
            delta_winlen = feaconf['delta_winlen']
            features = beer.features.add_deltas(features,
                [delta_winlen] * delta_order)

        # Mean normalization.
        if feaconf['utt_mnorm']:
            features -= features.mean(axis=0)[None, :]

        # Store the features as a numpy file.
        path = os.path.join(args.outdir, uttid)
        np.save(path, features)


if __name__ == '__main__':
    main()

