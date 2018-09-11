
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
    feats = np.load(feature_file)['features']
    dim = feats.shape[1]
    tot_sum = np.zeros(dim)
    tot_square_sum = np.zeros(dim)
    tot_counts = 0
    per_square_sum = (feats ** 2).sum(axis=0)
    tot_sum += feats.sum(axis=0)
    tot_square_sum += per_square_sum
    tot_counts += len(feats)
    #mean = tot_sum / tot_counts
    #var = tot_square_sum / tot_counts - mean ** 2
    return tot_sum, tot_square_sum, int(tot_counts)

def main():
    parser = argparse.ArgumentParser(description='Accumulate global data \
        statistics: mean, variance and frames')
    parser.add_argument('batches', type=str, help='list of feature file')
    parser.add_argument('stats', type=str, help='Feature statistics')
    args = parser.parse_args()
    data_stats = args.stats

    mean = var = tot_counts = None
    with open(args.batches, 'r') as f:
        for line in f:
            bmean, bvar, btot_counts = accumulate(line.strip())
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
