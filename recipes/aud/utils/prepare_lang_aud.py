'''Prepare Aucoustic Unit Discovery system "lang" directory.'''
import os
import argparse


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--non-speech-unit', action='store_true',
                        help='add a non-speech unit named "sil"')
    parser.add_argument('nunits', type=int, help='Maximum number of acoustic units' )
    #parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    if args.non_speech_unit:
        print('sil', 'non-speech-unit')

    for i in range(1, args.nunits + 1):
        print('au' + str(i), 'speech-unit')


if __name__ == '__main__':
    main()

