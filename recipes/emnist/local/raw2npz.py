'''Convert the raw MNIST format to a list of npz archives.'''

import argparse
import os
import struct

import numpy as np

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='batch size to split the data.')
    parser.add_argument('images', help='path to the raw images')
    parser.add_argument('labels', help='path to the raw labels')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    # Read the labels.
    with open(args.labels, 'rb') as fid:
        _, num = struct.unpack(">II", fid.read(8))
        labels = np.fromfile(fid, dtype=np.int8)

    # Read the images.
    with open(args.images, 'rb') as fid:
        magic, num, rows, cols = struct.unpack(">IIII", fid.read(16))
        images = np.fromfile(fid, dtype=np.uint8).reshape(num, rows, cols)

    # For safety.
    assert len(labels) == len(images)

    if args.batch_size <= 0:
        bsize = len(labels)
    else:
        bsize = args.batch_size

    # Normalizing constant to set all the features dimension between 0
    # and 1.
    norm_const = float(np.iinfo(np.uint8).max)

    idx = 0
    batchno = 1
    while idx < len(images):
        batch_images = images[idx: idx + bsize].astype(np.float32)
        batch_images = batch_images.reshape(len(batch_images), -1)
        batch_images /= norm_const
        batch_images = np.random.binomial(1, batch_images)
        batch_labels = labels[idx: idx + bsize].astype(np.int32) - 1
        outpath = os.path.join(args.outdir, 'batch' + str(batchno))
        np.savez(outpath, features=batch_images, labels=batch_labels)
        idx += bsize
        batchno += 1


if __name__ == '__main__':
    run()
