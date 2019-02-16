
import os
import argparse


def main():
    parser = argparse.ArgumentParser('Prepare phones files')
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('phonelist', help='phone.60-48-39.map')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    phone_48_file = os.path.join(args.outdir, 'units')
    phone_39_file = os.path.join(args.outdir, 'phones_39.txt')
    phone_map_file = os.path.join(args.outdir, 'phones_48_to_39.txt')

    dict_phone_47 = {}
    dict_map = {}
    with open(args.phonelist, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] != 'q' and tokens[1] != 'sil':
                p_47 = tokens[1]
                p_39 = tokens[2]
                dict_phone_47[p_47] = 0
                dict_map[p_47] = p_39
    phone_39 = sorted(set(dict_map.values()))
    phone_39.remove('sil')
    phone_39 = ['sil'] + phone_39
    dict_39_id = dict((j, i) for i, j in enumerate(phone_39))

    with open(phone_48_file, 'w') as f1, open(phone_map_file, 'w') as f2:
            print('sil non-speech-unit', file=f1)
            print('sil sil', file=f2)
            for i, k in enumerate(sorted(dict_phone_47.keys())):
                if k == 'epi':
                    print(k, 'non-speech-unit', file=f1)
                else:
                    print(k, 'speech-unit', file=f1)
                print(k, dict_map[k], file=f2)
    with open(phone_39_file, 'w') as f4:
        for k in dict_39_id.keys():
            print(k, dict_39_id[k], file=f4)

if __name__ == '__main__':
    main()

