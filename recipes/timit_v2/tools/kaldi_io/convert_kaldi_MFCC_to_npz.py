
''' Read from stdin: output of Kaldi command copy-feats.
    Write converted Kaldi MFCC features into npz files.
'''

import sys
import numpy as np
import re

def main():
    outdir = sys.argv[1]
    instream = sys.stdin.read()
    uttStr = re.sub('(\[|\])', '', instream)
    uttList= uttStr.split('\n')
    uttList = list(filter(None, uttList))
    uttName = re.sub(r'\s+', '', uttList.pop(0))
    print(uttName)
    arr= []
    for nk, k in enumerate(uttList):
        perFrame = list(map(float, k.split()))
        arr.append(perFrame)
    arrNp = np.array(arr)
    np.save(outdir + uttName + '.npy', arrNp)

if __name__ == '__main__':
    main()
