'Compute the normalized mutual information.'

import argparse
import yaml
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', action='store_true',
                        help='old normalized version')
    parser.add_argument('counts', help='YAML file containing the counts')
    args = parser.parse_args()

    with open(args.counts, 'r') as f:
        au2phone = yaml.load(f)

    # Extract the list of acoustic units / phones.
    au_set = set()
    phone_set = set()
    for au in au2phone:
        au_set.add(au)
        for phone in au2phone[au]:
            phone_set.add(phone)
    au_set = list(au_set)
    phone_set = list(phone_set)

    # Build the counts matrix.
    counts = np.zeros((len(au_set), len(phone_set))) + 1e-37
    for au in au2phone:
        for phone in au2phone[au]:
            au_idx, phone_idx = au_set.index(au), phone_set.index(phone)
            counts[au_idx, phone_idx] += au2phone[au][phone]

    p_X = counts.sum(axis=-1) / counts.sum()
    p_Y = counts.sum(axis=0) / counts.sum()
    p_X_Y = counts / counts.sum()
    p_Y_given_X = counts / counts.sum(axis=1, keepdims=True)
    p_X_given_Y = counts / counts.sum(axis=0, keepdims=True)

    H_X = - p_X @ np.log2(p_X + 1e-37)
    H_Y = - p_Y @ np.log2(p_Y + 1e-37)
    H_Y_given_X = - np.sum(p_X_Y * np.log2(p_Y_given_X))
    H_X_given_Y = - np.sum(p_X_Y * np.log2(p_X_given_Y))

    print(f'# units: {len(p_X)}')
    print('NMI (%)')
    if args.old:
        print(f'{100 * (H_Y - H_Y_given_X) / H_Y:.2f}')
    else:
        print(f'{200 * (H_Y - H_Y_given_X) / (H_Y + H_X):.2f}')


if __name__ == '__main__':
    main()

