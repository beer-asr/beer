'''Compute the false positive and false negative rates to plot DET curve 
   for model selection task.
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

def read_details(fid):
    dict_utt = {}
    with open(fid, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            dict_utt[tokens[0]] = float(tokens[1])
    return dict_utt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thres', type=str, action='store', help='Thresholds')
    parser.add_argument('uttid_true',
        help='List of correct reference utterence ids')
    parser.add_argument('uttid_false',
        help='List of wrong reference utterence ids')
    parser.add_argument('details', help='File with per utterance details.')
    parser.add_argument('output', help='Output file with false negatve rate\
        and false postive rate')
    args = parser.parse_args()

    thres = [float(i) for i in args.thres.split(',')]
    true_utts = read_uttids(args.uttid_true)
    false_utts = read_uttids(args.uttid_false)
    details = read_details(args.details)
    output = args.output
    with open(output, 'w') as fid:
        for t in thres:
            fn = 0
            fp = 0
            for k in list(details.keys()):
                diff = details[k]
                if (diff < t) and (k in false_utts):
                    fn += 1
                if (diff > t) and (k in true_utts):
                    fp += 1
            print('Threshold: {0:.2f}'.format(t),
                  'FN: {0:.2f}'.format(fn/len(false_utts)),
                  'FP: {0:.2f}'.format(fp/len(true_utts)), file=fid)

if __name__ == '__main__':
    main()
