
import os
import argparse


def main():
    parser = argparse.ArgumentParser('Prepare phones files')
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('phonelist', help='phone.60-48-39.map')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    phone_48 = args.outdir + '/phones_48.txt'
    phone_map = args.outdir + '/phones_48_to_39.txt'

    dict_phone_48 = {}
    dict_map = {}
    with open(args.phonelist, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] != 'q':
                p_48 = tokens[1]
                p_39 = tokens[2]
            dict_phone_48[p_48] = 0
            dict_map[p_48] = p_39

    with open(phone_48, 'w') as f1, open(phone_map, 'w') as f2:
        f1.write('sil 0\n')
        f2.write('sil sil\n')
        for i, k in enumerate(sorted(dict_map.keys())):
            if k != 'sil':
                f1.write(k + ' ' + str(i+1) + '\n')
                f2.write(k + ' ' + dict_map[k] + '\n')

if __name__ == '__main__':
    main()

