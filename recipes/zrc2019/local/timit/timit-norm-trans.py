
'''Normalize the TIMIT transcription by mapping the set of phones
to a smaller subset.

'''

import argparse
import sys
import logging


logging.basicConfig(format='%(levelname)s: %(message)s')


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--map-60-48', action='store_true')
    group.add_argument('--map-48-39', action='store_true')
    parser.add_argument('phonemap', help='the 60-48-39 mapping')
    args = parser.parse_args()

    # Load the phone map.
    map_60to48 = {}
    map_48to39 = {}
    to_remove = []
    with open(args.phonemap, 'r') as fid:
        for line in fid:
            phones = line.strip().split()
            try:
                map_60to48[phones[0]] = phones[1]
                map_48to39[phones[1]] = phones[2]
            except IndexError:
                to_remove.append(phones[0])

                # If there is no mapping for a phone else than "q"
                # print a warning message.
                if not phones[0] == 'q':
                    msg = 'No mapping for the phone "{}". It will be ' \
                          'removed from the transcription.'
                    logging.warning(msg.format(phones[0]))

    # Select the requested mapping from the command line arguments.
    if args.map_60_48:
        mapping = map_60to48
    else:
        mapping = map_48to39

    # Normalize the transcription
    for line in sys.stdin:
        tokens = line.strip().split()
        uttid = tokens[0]
        utt_trans = tokens[1:]

        # Remove the phones that have no mapping from the
        # original transcription.
        utt_trans = [phone for phone in utt_trans
                     if phone not in to_remove]

        new_utt_trans = map(lambda x: mapping[x], utt_trans)
        print(uttid, ' '.join(new_utt_trans))


if __name__ == '__main__':
    run()

