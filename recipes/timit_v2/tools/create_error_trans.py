
import argparse
import numpy as np


def read_phone(fid):
    phones = []
    with open(fid, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            phones.append(tokens[0])
    return phones

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('precision', type=float, help='Percentage of \
        transcription correctness')
    parser.add_argument('phonelist', help='File with 48 phones: phones.txt')
    parser.add_argument('input_trans', help='Original transcription')
    parser.add_argument('output_trans', type=str, help='Artifitially modified \
        transcriptions with errors')
    args = parser.parse_args()

    error_rate = (1 - args.precision) / 3
    phones = read_phone(args.phonelist)
    np.random.seed(1000000)
    with open(args.input_trans, 'r') as f, open(args.output_trans, 'w') as o:
        for line in f:
            tokens = line.strip().split()
            utt = tokens.pop(0)
            new_tokens = []
            for i, j in enumerate(tokens):
                choice = np.random.choice(['del', 'sub', 'ins', 'true'], 
                         p=[error_rate, error_rate, error_rate, args.precision])
                if (choice == 'true') or (j == 'sil'):
                    new_tokens.append(j)
                elif choice == 'sub':
                    sub_phones = phones.copy()
                    sub_phones.remove(j)
                    random_phone = np.random.choice(sub_phones,
                                   p=np.ones(len(sub_phones))/len(sub_phones))
                    new_tokens.append(random_phone)
                elif choice == 'ins':
                    random_phone = np.random.choice(phones,
                                   p=np.ones(len(phones))/len(phones))
                    new_tokens.extend((j, random_phone))
            print(utt, ' '.join(new_tokens), file=o)


if __name__ == "__main__":
    main()
