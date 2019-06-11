
import argparse
import os


def read_timit_labels(path, frate=100, srate=16000):
    '''Read TIMIT label files.

    Args:
        path (str): Path to the TIMIT label file.
        samp_period (int): Features sampling rate in
            100ns (default is 100 Hz).
        srate (int): Audion sampling rate (default: 16000 Hz)

    Returns:
       list of tuple

    '''
    factor = frate / srate
    segmentation = []
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            if len(tokens) != 3:
                raise ValueError('File is badly formatted.')
            start = int(int(tokens[0]) * factor)
            end = int(int(tokens[1]) * factor)
            segmentation += [tokens[2]] * (end - start)
    return segmentation


def main():
    parser = argparse.ArgumentParser('Exract phone alignment from TIMIT label files')
    parser.add_argument('--frate', default=100, type=int,
                        help='frame rate (Hz) of the alignment (default: 100)')
    parser.add_argument('--srate', default=16000, type=int,
                        help='audio rate (Hz) (default: 16000)')
    parser.add_argument('labelfiles', help='list of label files')
    args = parser.parse_args()

    with open(args.labelfiles, 'r') as f:
        for line in f:
            path = line.strip()
            fname = os.path.basename(path)
            root, ext = os.path.splitext(fname)
            spkname = os.path.basename(os.path.dirname(path))
            uttid = spkname + '_' + root
            trans = read_timit_labels(path, args.frate, args.srate)
            print(uttid, ' '.join(trans))


if __name__ == '__main__':
    main()

