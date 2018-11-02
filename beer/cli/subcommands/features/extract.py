'extract speech features from a list of wav files'


import argparse
import beer
import io
import os
import subprocess
import sys

import yaml
import numpy as np
from scipy.io.wavfile import read


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
    'utt_mnorm': False,
    'add_energy': True,
}


def compute_dct_bases(nfilters, n_dct_coeff):
    dct_bases = np.zeros((nfilters, n_dct_coeff))
    for m in range(n_dct_coeff):
        dct_bases[:, m] = np.cos((m+1) * np.pi / nfilters * (np.arange(nfilters) + 0.5))
    return dct_bases


class ShowDefaultsAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        print(yaml.dump(feaconf, default_flow_style=False), end='')
        parser.exit()


def setup(parser):
    parser.add_argument('--show-default-conf', action=ShowDefaultsAction,
                        help='show the default configuration and exit')
    parser.add_argument('feaconf', help='configuration file of the '
                                        'features')
    parser.add_argument('wav_list', help='list of WAV files or "-" for stdin')
    parser.add_argument('outdir', help='output directory')


def main(args, logger):

    # Override the default configuration.
    with open(args.feaconf, 'r') as fid:
        new_conf = yaml.load(fid)

    # Check for unknown options.
    for key in new_conf:
        if key not in feaconf:
            logger.error('Unknown setting "{}"'.format(key))
            exit(1)
    feaconf.update(new_conf)

    # Pre-compute the DCT bases.
    dct_bases = compute_dct_bases(feaconf['nfilters'], feaconf['n_dct_coeff'])


    if args.wav_list == '-':
        infile = sys.stdin
    else:
        with open(args.wav_list, 'r') as f:
            infile = f.readlines()

    counts = 0
    for line in infile:
        tokens = line.strip().split()
        uttid, inwav = tokens[0], ' '.join(tokens[1:])
        logger.debug(f'processing utterance: {uttid}')

        # If 'inwav' ends up with the '|' symbol, 'inwav' is
        # interpreted as a command otherwise we assume 'inwav' to
        # be a path to a wav file.
        if inwav[-1] == '|':
            cmd = inwav[:-1]
            logger.debug(f'reading command: {cmd}')
            proc = subprocess.run(cmd, shell=True,
                                  stdout=subprocess.PIPE)
            sr, signal = read(io.BytesIO(proc.stdout))
        else:
            logger.debug(f'reading file: {inwav}')
            sr, signal = read(inwav)
        if not sr == feaconf['srate']:
            msg = 'Sampling rate ({}) does not match the one ' \
                  'of the given file ({}).'
            logger.error(msg.format(feaconf['srate'], sr))
            exit(1)

        # Mel spectrum.
        logger.debug('extracting STFT')
        melspec, fft_len = beer.features.short_term_mspec(
            signal,
            flen=feaconf['window_len'],
            frate=feaconf['framerate'],
            preemph=feaconf['preemph'],
            srate=feaconf['srate'],
        )

        # Filter bank.
        if feaconf['apply_fbank']:
            logger.debug(f'applying filter bank (F={feaconf["nfilters"]})')
            fbank = beer.features.create_fbank(feaconf['nfilters'], fft_len,
                                               lowfreq=feaconf['cutoff_lfreq'],
                                               highfreq=feaconf['cutoff_hfreq'])
            melspec = melspec @ fbank.T

        # Take the logarithm of the magnitude spectrum.
        logger.debug('log of the STFT')
        log_melspec = np.log(1e-6 + melspec)

        # HTK compatibility normalization (probably doesn't change
        # the accuracy of the recognition).
        norm = np.sqrt(2. / feaconf['nfilters'])

        # DCT transform.
        if feaconf['apply_dct']:
            logger.debug('cosine transform of the log STFT')
            features = log_melspec @ dct_bases

            features *= norm

            # Liftering.
            logger.debug('cepstrum liftering')
            l_coeff = feaconf['lifter_coeff']
            lifter = 1 + (l_coeff / 2) * np.sin(np.pi * \
                (1 + np.arange(feaconf['n_dct_coeff'])) / l_coeff)
            features *= lifter
        else:
            features = log_melspec

        # Deltas.
        if feaconf['apply_deltas']:
            logger.debug('concatenating derivatives')
            delta_order = feaconf['delta_order']
            delta_winlen = feaconf['delta_winlen']
            features = beer.features.add_deltas(features,
                tuple([delta_winlen] * delta_order))

        if feaconf['add_energy']:
            logger.debug('add the energy to the features')
            energy = log_melspec.sum(axis=-1) * norm
            features = np.c_[energy, features]

        # Mean normalization.
        if feaconf['utt_mnorm']:
            logger.debug('utterance mean normalization')
            features -= features.mean(axis=0)[None, :]

        # Store the features as a numpy file.
        path = os.path.join(args.outdir, uttid)
        logger.debug(f'saving features to: {path}')
        np.save(path, features)

        counts += 1

    logger.info(f'extracted features for {counts} file(s)')


if __name__ == '__main__':
    main()

