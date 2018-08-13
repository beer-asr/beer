#!/usr/bin/python3


import sys
import numpy as np
import argparse
import re
import subprocess


def readScp(scpfile):
    '''Readin: Kaldi MFCC scp file, each line is an utterance 
    Return: Dict. Key is uttname(str), value is MFCC for this utt( frame x 13 np array)
    '''

    uttDict = {}
    with open(scpfile, 'r') as s:
        files = s.readlines()
        for i, j in enumerate(files):
            output = subprocess.check_output\
            ("/export/b07/jyang/kaldi-jyang/kaldi/src/featbin/copy-feats scp:\"awk \'NR=={count}\' {f} |\" ark,t:-".format(count=i+1, f=scpfile), shell=True)
            uttStr = output.decode('utf-8')
            uttStr = re.sub('(\[|\])', '', uttStr)
            uttList= uttStr.split('\n')
            uttList = list(filter(None, uttList))
            uttName = re.sub(r'\s+', '', uttList.pop(0))
            arr= []
            for nk, k in enumerate(uttList):
                perFrame = list(map(float, k.split()))
                arr.append(perFrame)
            arrNp = np.asarray(arr)
            uttDict[uttName] = arrNp
    return uttDict


#def main():
#    parser = argparse.ArgumentParser\
#    (description = 'Conver Kaldi MFCC features into python np arrays')
#    parser.add_argument('list', help = 'Input MFCC scp file')
#    args = parser.parse_args()
#    for k in readScp(args.list).keys():
#        print (readScp(args.list)[k].shape)

#if __name__ == '__main__':
#    main()

