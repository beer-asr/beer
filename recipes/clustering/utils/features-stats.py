
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='Accumulate global data \
        statistics: mean, variance and frames')
    parser.add_argument('features', type=str, help='list of feature file')
    parser.add_argument('stats', type=str, help='Feature statistics')
    args = parser.parse_args()
    data_stats = args.stats

    mean = var = tot_counts = None
    features = np.load(args.features)
    for utterance in features.files:
        feats = features[utterance]
        bmean = feats.sum(axis=0)
        bvar = (feats**2).sum(axis=0)
        btot_counts = len(feats)
        if mean is None:
            mean, var, tot_counts = bmean, bvar, btot_counts
        else:
            mean += bmean
            var += bvar
            tot_counts += btot_counts

    mean = mean / tot_counts
    var = var / tot_counts - mean ** 2
    stats = {'mean': mean, 'var': var, 'nframes': tot_counts}
    np.savez(data_stats, **stats)

if __name__ == "__main__":
    main()
