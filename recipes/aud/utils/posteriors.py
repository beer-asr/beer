
'Compute the HMM state posteriors.'

# I included a scale parameters as you probably want to "smooth" the
# posteriors to get better results. If the acoustic scale is 0 then
# the output posteriors will be completely flat (= 1/K). On the other
# hand if the acoustic scale is large the posteriors will be very
# "sharp" (i.e. one value will be 1 and the other 0). I think good
# values for the scale parameter should be less than 1 something
# around 0.3 or 0.2 should be a good start (as I remember Kaldi uses
# 1/12 by default but they have a strong language model).

import argparse
import os
import pickle
import sys

import numpy as np
import beer


def compute_posts(model, data, scale):
    inference_graph = model.graph.value
    stats = model.sufficient_statistics(data)
    pc_llhs = model._pc_llhs(stats, inference_graph) * scale
    return model._inference(pc_llhs, inference_graph)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--acoustic-scale', default=1., type=float,
                        help='scaling factor of the acoustic model')
    parser.add_argument('model', help='hmm based model')
    parser.add_argument('dataset', help='training data set')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    scale = args.acoustic_scale
    for utt in dataset.utterances(random_order=False):
        print(f'processing utterance: {utt.id}')
        posts = compute_posts(model, utt.features, scale).detach().numpy()

        # posts is a NxK matrix where N is the number of frames and K is
        # the total number of HMM states (K = n_units x n_states_per_unit).
        # To get the posterior per units you can do something like:
        # posts = posts.reshape(len(posts), n_units, -1).sum(axis=-1)

        path = os.path.join(args.outdir, f'{utt.id}.npy')
        np.save(path, posts)


if __name__ == "__main__":
    main()

