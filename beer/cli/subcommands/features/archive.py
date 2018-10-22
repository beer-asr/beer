
'create an archive from a features directory'

import argparse
import glob
import os
import pickle
from zipfile import ZipFile


def setup(parser):
    parser.add_argument('-e', '--extension', default='npy',
                        help='extension of the features file (default: npy)')
    parser.add_argument('feadir', help='features directory')
    parser.add_argument('out', help='output zip archived')


def main(args, logger):
    counts = 0
    with ZipFile(args.out, 'w') as f:
        for path in glob.glob(os.path.join(args.feadir, '*' + args.extension)):
            logger.debug(f'adding {path} to the archive')
            arcname = os.path.basename(path).replace('.' + args.extension, '')
            f.write(path, arcname=arcname)
            counts += 1
    logger.info(f'created archive from {counts} features files')

if __name__ == "__main__":
    main()

