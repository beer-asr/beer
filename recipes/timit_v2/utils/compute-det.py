'''Compute the false positive and false negative rates to plot DET curve 
   for model selection task.
   Given the likelihood from both models and a threshold, if the
   variance of difference(log domain) between two llhs is above threshold, 
   then decalare a positive detection.
'''

import argparse
import numpy as np

def read_uttids(uttlist):
    dict_utt = {}
    with open(uttlist, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            dict_utt[tokens[0]] = 0
    return dict_utt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thres', type=str, action='store', help='Thresholds')
    parser.add_argument('uttid_true',
        help='List of correct reference utterence ids')
    parser.add_argument('uttid_false',
        help='List of wrong reference utterence ids')
    parser.add_argument('ali_llhs', help='Log-likelihood from the alignments')
    parser.add_argument('hyp_llhs', help='Log-likelihood from the acoustic \
        model with no reference')
    parser.add_argument('result', help='Output file with false negatve rate\
        and false postive rate')
    args = parser.parse_args()

    thres = [float(i) for i in args.thres.split()]
    true_utts = read_uttids(args.uttid_true)
    false_utts = read_uttids(args.uttid_false)

    ali_llhs = np.load(args.ali_llhs)
    hyp_llhs = np.load(args.hyp_llhs)
    with open(args.result, 'w') as fid:
        for t in thres:
            fn = 0
            fp = 0
            for k in list(ali_llhs.keys()):
                diff = ali_llhs[k] - hyp_llhs[k]
                diff = diff.var()
                if (diff < t) and (k in false_utts):
                    fn += 1
                if (diff > t) and (k in true_utts):
                    fp += 1
            print(fp, fn)
            print('Threshold: {0:.2f}'.format(t),
                  'FN: {0:.2f}'.format(fn/len(false_utts)),
                  'FP: {0:.2f}'.format(fp/len(true_utts)), file=fid)

if __name__ == '__main__':
    main()
