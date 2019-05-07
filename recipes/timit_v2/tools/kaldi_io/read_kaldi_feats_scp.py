#!/usr/bin/python3


import sys
import os
import re
import subprocess
import numpy as np

copy_feat = "/export/b07/jyang/kaldi-jyang/kaldi/src/featbin/copy-feats"

def main():
    scpfiles = sys.argv[1] # Input scpfiles
    outdir = sys.argv[2] # Target directory
    with open(scpfiles, 'r') as scp:
        for i, j in enumerate(scp.readlines()):
            cmd = f'echo \"{j}\" | {copy_feat} scp:- ark,t:-'
            output = subprocess.check_output(cmd, shell=True)
            uttStr = output.decode('utf-8')
            uttStr = re.sub('(\[|\])', '', uttStr)
            uttList= uttStr.split('\n')
            uttList = list(filter(None, uttList))
            uttName = re.sub(r'\s+', '', uttList.pop(0))
            arr= []
            for nk, k in enumerate(uttList):
                perFrame = list(map(float, k.split()))
                arr.append(perFrame)
            fname = os.path.join(outdir, uttName + '.npy')
            np.save(fname, np.array(arr))

if __name__ == '__main__':
    main()

