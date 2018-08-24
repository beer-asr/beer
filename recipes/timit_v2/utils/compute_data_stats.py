
import numpy as np
import argparse

def accumulate(feature_file):
    '''Compute global mean, variance, frame counts
    Argument:
        feature_file(str): feature file(npz)
    Returns:
        mean: np array (float)
        var: np array (float)
        tot_counts(int): total frames in feature files
    '''
    feats = np.load(feature_file)
    keys = list(feats.keys())
    dim = feats[keys[0]].shape[1]
    tot_sum = np.zeros(dim)
    tot_square_sum = np.zeros(dim)
    tot_counts = 0
    for k in keys:
        nframes_per_utt = len(feats[k])
        per_square_sum = (feats[k] ** 2).sum(axis=0)
        tot_sum += feats[k].sum(axis=0)
        tot_square_sum += per_square_sum
        tot_counts += nframes_per_utt
    mean = tot_sum / tot_counts
    var = tot_square_sum / tot_counts - mean ** 2
    return mean, var, int(tot_counts)

def main():
    parser = argparse.ArgumentParser(description='Accumulate global data \
        statistics: mean, variance and frames')
    parser.add_argument('features', type=str, help='Feature file')
    parser.add_argument('stats', type=str, help='Feature statistics')
    args = parser.parse_args()
    feats = args.features
    data_stats = args.stats
    mean, var, tot_counts = accumulate(feats)
    stats = {'mean': mean, 'var': var, 'nframes': tot_counts}
    np.savez(data_stats, **stats)

if __name__ == "__main__":
    main()
