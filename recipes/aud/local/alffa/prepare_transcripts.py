#!/usr/bin/env python3

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang',
                        help='language name, will be prepended to phone names')
    parser.add_argument('--keep-sil-phones', action='store_true',
                        help='do not filter out phones that do not exist in nonsilence_phones.txt')
    parser.add_argument('text',
                        help='input text file containing sentence transcripts')
    parser.add_argument('lexicon',
                        help='map from word to sequence of phones')
    parser.add_argument('output_dir',
                        help='output directory into which to store the transcripts')

    args = parser.parse_args()
    langdir = os.path.join(args.output_dir, '../lang/')
    phone_to_id_file = os.path.join(langdir, 'phones.txt')
    lexicon = {sent.strip().split()[0]: sent.strip().split()[1:]
               for sent in open(args.lexicon, encoding='utf-8')}

    for di in [langdir, args.output_dir]:
        if not os.path.isdir(di):
            os.makedirs(di)

    phone_to_id = {}
    if not os.path.isfile(phone_to_id_file):
        for sent in lexicon.values():
            for phone in sent:
                if phone not in phone_to_id.keys():
                    phone_to_id[phone] = '_'.join([args.lang, str(len(phone_to_id) + 1)])
        with open(phone_to_id_file, 'w', encoding='utf-8') as _ph:
            for k, v in phone_to_id.items():
                _ph.write(f'{k} {v}\n')
    else:
        with open(phone_to_id_file, encoding='utf-8') as _ph:
            for line in _ph:
                ln = line.strip().split()
                phone_to_id[ln[0]] = ln[1]

    nonsil_phones_file = os.path.join(os.path.dirname(args.lexicon), 'nonsilence_phones.txt')
    if os.path.isfile(nonsil_phones_file):
        nonsilphones = [phone.strip() for phone in open(nonsil_phones_file, encoding='utf-8')]
    else:
        nonsilphones = [phone for phone in phone_to_id.keys()]
    word_to_phone_ids = {}
    transcripts = {sent.strip().split()[0]: sent.strip().split()[1:]
                   for sent in open(args.text, encoding='utf-8')}
    with open(os.path.join(args.output_dir, 'trans'), 'w', encoding='utf-8') as _out:
        for utt, trans in transcripts.items():
            print_utt = True
            phone_id_trans = [utt, 'sil']
            for word in trans:
                try:
                    phone_trans = lexicon[word]
                    if not args.keep_sil_phones:
                        phone_trans = [phone for phone in phone_trans if phone in nonsilphones]
                except KeyError:
                    phone_trans = '_'.join([args.lang, str(0)])
                try:
                    phone_id_trans += [phone_to_id[phone] for phone in phone_trans]
                except KeyError:
                    #print(utt)
                    print_utt = False
                    break               
            phone_id_trans += ['sil']
            if print_utt:
                _out.write(f'{" ".join(phone_id_trans)}\n')
    
                    
if __name__ == '__main__':
    main()
