
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diff-detail', default='var',
        help='Compute the absolute value or variance of llhs difference')
    parser.add_argument('ali_llhs', help='Alignment log-likelihoods')
    parser.add_argument('hyp_llhs', help='Decode log-likelihoods')
    parser.add_argument('output')
    args = parser.parse_args()

    ali_llhs = np.load(args.ali_llhs)
    hyp_llhs = np.load(args.hyp_llhs)

    with open(args.output, 'w') as f:
        for k in ali_llhs.keys():
            print(k, (ali_llhs[k] - hyp_llhs[k]).var(), file=f)

if __name__ == "__main__":
    main()
