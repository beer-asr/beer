
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

def corrupt_one_utterance(tokens, error_type, error_prob,
                          sub_map=None, ins_list=None):
    new_tokens = []
    error_prob = float(error_prob)
    if error_type == 'sub':
        for i, j in enumerate(tokens):
            choice = np.random.choice(['sub', 'true'],
                    p=[error_prob, 1 - error_prob])
            if choice == 'true':
                new_tokens.append(j)
            else:
                new_tokens.append(sub_map[j])
    elif error_type == 'del':
        for i, j in enumerate(tokens):
            choice = np.random.choice(['del', 'true'],
                    p=[error_prob, 1 - error_prob])
            if choice == 'true':
                new_tokens.append(j)
    elif error_type == 'ins':
        for i, j in enumerate(tokens):
            choice = np.random.choice(['ins', 'true'],
                    p=[error_prob, 1 - error_prob])
            if choice == 'true':
                new_tokens.append(j)
            else:
                random_word = np.random.choice(ins_list,
                            p=np.ones(len(ins_list))/len(ins_list))
                new_tokens.extend((j, random_word))
    elif error_type == 'all':
        del_prob = float(error_prob) * .4
        ins_prob = float(error_prob) * .3
        sub_prob = float(error_prob) * .3
        true_prob = 1 - del_prob - ins_prob - sub_prob
        for i, j in enumerate(tokens):
            choice = np.random.choice(['del', 'ins', 'sub', 'true'],
                 p=[del_prob, ins_prob, sub_prob, true_prob])
            if choice == 'true':
                new_tokens.append(j)
            elif choice == 'ins':
                random_word = np.random.choice(ins_list,
                            p=np.ones(len(ins_list))/len(ins_list))
                new_tokens.extend((j, random_word))
            elif choice == 'sub':
                new_tokens.append(sub_map[j])
    return new_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_type', choices=['del', 'sub', 'ins', 'all'])
    parser.add_argument('--error_prob', help='Total phone error rate')
    parser.add_argument('--ser', default=0.3,
                        help='Percentage of corrupted utterance')
    parser.add_argument('--sub_map', default=None, help='File contains \
                        word-to-be-subbed sub')
    parser.add_argument('--ins_list', default=None,
                        help='File contains list of words used for insertion')
    parser.add_argument('input_trans')
    parser.add_argument('output_trans')
    args = parser.parse_args()

    if args.error_type in ['sub', 'all']:
        if args.sub_map is None:
            sys.exit('Sublist should not be empty !')
        sub_map = read_sub_list(args.sub_map)
    if args.error_type in ['ins', 'all']:
        if args.ins_list is None:
            sys.exit('Insertion list should not be empty !')
        ins_list = read_ins_list(args.ins_list)
    np.random.seed(100000)
    ser = float(args.ser)

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
                 new_tokens = corrupt_one_utterance(tokens,
                                args.error_type, args.error_prob,
                                sub_map=read_sub_list(args.sub_map),
                                ins_list=read_ins_list(args.ins_list))
                 print(utt, ' '.join(new_tokens), file=o)

if __name__ == "__main__":
    main()
