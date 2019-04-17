'''Convert a text transcription into a phonetic transcription.'''
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--add-sil', action='store_true',
                        help='add silence phone "sil" at the end/beginning of the utterance')
    parser.add_argument('lexicon',
                        help='word to phone mapping')
    parser.add_argument('text', help='text transcription' )
    args = parser.parse_args()

    with open(args.lexicon, 'r') as f:
        lexicon = {}
        for line in f:
            tokens = line.strip().split()
            lexicon[tokens[0]] = ' '.join(tokens[1:])

    with open(args.text, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            uttid = tokens[0]
            try:
                phone_trans = ' '.join([lexicon[word] for word in tokens[1:]])
            except KeyError as err:
                print(f'skipping utterance: {uttid} because of missing word: {err}',
                      file=sys.stderr)
            if not args.add_sil:
                print(uttid, phone_trans)
            else:
                print(uttid, 'sil', phone_trans, 'sil')


if __name__ == '__main__':
    main()

