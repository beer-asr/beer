
import argparse
import numpy as np


def read_sub_list(fid):
    dict_phones = {}
    with open(fid, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            dict_phones[tokens[0]] = tokens[1]
    return dict_phones

def read_ins_list(fid):
    phones = []
    with open(fid, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            phones.append(tokens[0])
    return phones

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--del_ins_true_rate', required=True,
                        help='Prob of del, ins, true')
    parser.add_argument('--ser', default=0.4,
                        help='Percentage of corrupted utterance')
    parser.add_argument('sub_list', help='File contains \
                        word-to-be-subbed sub')
    parser.add_argument('insert_list', help='File contains list of words used\
                        for insertion')
    parser.add_argument('input_trans')
    parser.add_argument('output_trans')
    args = parser.parse_args()

    sub_dict = read_sub_list(args.sub_list)
    ins_list = read_ins_list(args.insert_list)
    del_prob = float(args.del_ins_true_rate.split()[0])
    ins_prob = float(args.del_ins_true_rate.split()[1])
    true_prob = float(args.del_ins_true_rate.split()[2])
    ser = float(args.ser)
    np.random.seed(100000)
    with open(args.input_trans, 'r') as f, open(args.output_trans, 'w') as o:
        for line in f:
            line = line.strip()
            choice_utt = np.random.choice(['corrupt', 'not_corrupt'],
                         p=[ser, 1-ser])
            if choice_utt == 'not_corrupt':
                print(line, file=o)
            else:
                 tokens = line.split()
                 utt = tokens.pop(0)
                 new_tokens = []
                 for i, j in enumerate(tokens):
                     choice = np.random.choice(['del', 'ins', 'true'],
                              p=[del_prob, ins_prob, true_prob])
                     if j in sub_dict.keys():
                         new_tokens.append(sub_dict[j])
                     elif choice == 'true':
                         new_tokens.append(j)
                     elif choice == 'ins':
                         random_word = np.random.choice(ins_list,
                                       p=np.ones(len(ins_list))/len(ins_list))
                         new_tokens.extend((j, random_word))
                 print(utt, ' '.join(new_tokens), file=o)

if __name__ == "__main__":
    main()
