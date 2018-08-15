
'Extract log Mel spectrum features.'


import argparse
import beer
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess


def main():
    parser = argparse.ArgumentParser('Extract Mel spectrum features.')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('--srate', type=int, default=16000,
                        help='expected sampling rate')
    parser.add_argument('--nfilters', type=int, default=30,
                        help='number of filters (30)')
    args = parser.parse_args()


    with open(args.scp, 'r') as fid:
        for line in fid:
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
            assert sr == args.srate, 'Input file has different sampling rate.'

            log_melspec = beer.features.fbank(signal, nfilters=args.nfilters,
                srate=args.srate)
            np.save(os.path.join(args.outdir, uttid), log_melspec)


if __name__ == '__main__':
    main()

