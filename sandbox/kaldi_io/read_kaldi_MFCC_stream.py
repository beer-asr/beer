#!/usr/bin/python3


import sys
import numpy as np
import argparse
import re
import subprocess
sys.path.append('/export/b07/jyang/kaldi-jyang/kaldi/src/featbin/')

def readStream(oneUtt):
    '''
    Readin: output of Kaldi command copy-feats, type str
    Return: Utt name (str) and feature matrix(numpy array, frame * 13 dimension)
    '''
    uttStr = re.sub('(\[|\])', '', oneUtt)
    uttList= uttStr.split('\n')
    uttList = list(filter(None, uttList))
    uttName = re.sub(r'\s+', '', uttList.pop(0))
    arr= []
    for nk, k in enumerate(uttList):
        perFrame = list(map(float, k.split()))
        arr.append(perFrame)
    arrNp = np.asarray(arr)
    return uttName, arrNp


#def main():
#    parser = argparse.ArgumentParser\
#    (description = 'Conver Kaldi MFCC features into python np arrays')
#    parser.add_argument('iList', help = 'Input MFCC scp file')
#    args = parser.parse_args()
#    with open(args.iList, 'r') as s:
#        filestr = s.read()
#        print (type(filestr))
#        uttName, arrNp = readScp(filestr)
#        print (arrNp.shape)

#if __name__ == '__main__':
#    main()

