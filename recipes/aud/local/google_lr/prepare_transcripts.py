#!/usr/bin/env python3

import os
import argparse
import unicodedata as ud

def _clean(x: str, diacritics=False) -> str:
    # puncs = ['-', '.', ',', '*', '/', '_']
    # mapper = {'ṛ': 'r', 'ś': 's', 'ạ': 'a', 'ẃ': 'w', 'ṕ': 'p'}
    if not diacritics:
        x = ''.join([a.lower() for a in x if a.isalnum()])
    else:
        x = ''.join([a.lower() for a in x if a.isalnum() or ud.combining(a)])
        # x = ''.join([a.lower() for a in x if a not in puncs])
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', required=True,
                        help='language name, will be prepended to phone names')
    parser.add_argument('--add-sil', action='store_true',
                        help='add the sil unit to the beginning and end of each utterance')
    parser.add_argument('--diacritics', action='store_true',
                        help='specify that the file has tonal diacritics')
    parser.add_argument('text',
                        help='input text file containing sentence transcripts')
    parser.add_argument('output_dir',
                        help='output directory into which to store the transcripts')

    args = parser.parse_args()
    langdir = os.path.join(args.output_dir, '../lang/')
    phone_to_id_file = os.path.join(langdir, 'phones.txt')

    for di in [langdir, args.output_dir]:
        if not os.path.isdir(di):
            os.makedirs(di)

    with open(args.text, encoding='utf-8') as f:
        trans = [line.strip().split() for line in f.readlines()]
    trans_dict = {line[0]: line[1:] for line in trans}
    phone_to_id = {'sil': 'sil'}
    N = {}
    with open(os.path.join(args.output_dir, 'trans'), 'w', encoding='utf-8') as _outfile:
        for utt, sent in trans_dict.items():
            t = []
            for word in sent:
                if word.startswith('['):
                    if t and t[-1] != 'sil':
                        t.append('sil')
                else:
                    word = _clean(word, args.diacritics)
                    if not word:
                        continue
                    phones = [word[0]]
                    for letter in word[1:]:
                        if ud.combining(letter):  # Diacritics
                            phones[-1] += letter
                            #if phones[-1] not in N:
                                # print(f'letter {phones[-1]} {ord(letter)} {len(N)}')
                                #N[phones[-1]] = 1
                            #else:
                                #N[phones[-1]] += 1
                        elif letter == 'b' and phones[-1] == 'g':  # gb sound
                            phones[-1] += letter
                            #elif letter == 'h' and phones[-1] == 'c':  # Not really Yoruba but exists e.g. China
                            #phones[-1] += letter
                            #elif letter == 'h' and phones[-1] == 's':  # Mislabeled e.g. Mitsubishi
                            #    phones[-1] = 'ṣ'
                            #elif phones[-1] == 'c':  # Also not really Yoruba, only UNESCO
                            #phones[-1] = 'k'
                        else:
                            phones.append(letter)
                    for phone in phones:
                        if phone in N.keys():
                            N[phone] += 1
                        else:
                            N[phone] = 1
                        phone = ''.join(sorted(phone))  # In case of unordered diacritics and undots
                        if phone not in phone_to_id.keys():
                            phone_to_id[phone] = f'{args.lang}_{len(phone_to_id)+1}'
                    word_in_phones = ' '.join([phone_to_id[''.join(sorted(phone))] for phone in phones])
                    t.append(word_in_phones)
            if args.add_sil:
                t = ['sil'] + t + ['sil']
            _outfile.write(f'{utt} {" ".join(t)}\n')

    with open(phone_to_id_file, 'w', encoding='utf-8') as _outfile:
        for k, v in phone_to_id.items():
            _outfile.write(f'{k} {v}\n')
    #for k, v in N.items():
    #    print(f'{k}: {v}')
                    
if __name__ == '__main__':
    main()
