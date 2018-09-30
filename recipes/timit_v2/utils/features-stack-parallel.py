'''Create new set of features by stacking adjacent frames to the
original one. The identifiers of the utterances to process are read from
the standard input. The output directory is assumed to exist.

The new dimension of the features will be: dim = orig_dim x (2 x context + 1)

Example:
  $ cat uttids | python {script} --context 5 fbank.npz tmp/
  $ find tmp -name '*npy' | zip -j -@ stacked_fbank.npz && rm -r tmp

'''

import argparse
import os
import sys

import numpy as np


def stack_features(data, context):
    if context == 0:
        return data
    padded = np.r_[np.repeat(data[0][None], context, axis=0), data,
                   np.repeat(data[-1][None], context, axis=0)]
    stacked_features = np.zeros((len(data), (2 * context + 1) * data.shape[1]))
    for i in range(context, len(data) + context):
        sfea = padded[i - context: i + context + 1]
        stacked_features[i - context] = sfea.reshape(-1)
    return stacked_features


def run():
    parser = argparse.ArgumentParser(description=__doc__.format(script=__file__),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('fea', help='original features')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--context', type=int, default=1,
                        help='number of context frame (one side) to stack')
    args = parser.parse_args()

    fea = np.load(args.fea)

    for line in sys.stdin:
        uttid = line.strip()
        utt_fea = fea[uttid]
        new_fea = stack_features(utt_fea, args.context)
        outpath = os.path.join(args.outdir, uttid + '.npy')
        np.save(outpath, new_fea)


if __name__ == '__main__':
    run()
