
import sys

srcfile = sys.argv[1]
outfile = sys.argv[2]

with open(srcfile, 'r') as f:
    with open(outfile, 'w') as d:
        for l in f:
            l = l.strip()
            l = l.split()
            if len(l) == 3:
                d.write(l[1] + ' ' + l[2] + '\n')

