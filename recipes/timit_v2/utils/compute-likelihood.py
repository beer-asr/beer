'''Compute per frame/unit level likelihood'''

import argparse
import sys
import pickle
import numpy as np
import torch
import beer

def read_txtfile(infile):
    dict_f = {}
    with open(infile, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            key = tokens.pop(0)
            dict_f[key] = tokens
    return dict_f

def read_pdf2phone(infile):
    mapping = {}
    start_pdfs = []
    prev_unit = ''
    with open(infile, 'r') as p:
        for line in p:
            tokens = line.strip().split()
            mapping[int(tokens[0])] = tokens[1]
            if tokens[1] != prev_unit:
                start_pdfs.append(tokens[0])
                prev_unit = tokens[1]
    return mapping, start_pdfs

def merge_llhs(llhs, pdf_seq, start_ids):
    if (len(llhs) != len(pdf_seq)):
        sys.exit('Merging log-likelihoods: length of likelihood not matching \
                 with best path !')
    phone_llhs = []
    llhs_sum = llhs[0]
    frame_count = 1
    for i in range(1, len(pdf_seq)):
        if (pdf_seq[i] != pdf_seq[i-1]) and (pdf_seq[i] in start_ids):
            phone_llhs.append(llhs_sum / frame_count)
            llhs_sum = 0.
            frame_count = 0
        llhs_sum += llhs[i]
        frame_count += 1
    return phone_llhs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Decoding model')
    parser.add_argument('feats', type=str, help='Features to process')
    parser.add_argument('outfile', type=str, help='Output file')
    parser.add_argument('--align', type=str,
        help='A npz file with per frame alignment. If not empty, use it \
        to merge per farme likelihood into phone level')
    parser.add_argument('--pdf2phone', type=str,
        help='File mapping pdf id(int) into phone symbols(str)')
    args = parser.parse_args()
    feats = np.load(args.feats)
    outfile = args.outfile
    alignfile = args.align
    mapfile = args.pdf2phone
    smooth = None
    print('main')
    if alignfile:
        if not mapfile:
            sys.exit('Pdf2phone mapping file needed if merging likelihood' +
                     'into phone level')
        else:
            aligns = np.load(alignfile)
            _, start_ids = read_pdf2phone(mapfile)
            smooth = 1
    with open(args.model, 'rb') as m:
        model = pickle.load(m)
    with open(outfile, 'w') as fid:
        for k in feats.keys():
            print(k)
            ft = torch.from_numpy(feats[k]).float()
            stats = model.sufficient_statistics(ft)
            llhs = model.expected_log_likelihood(stats).tolist()
            if smooth:
                ref_path = aligns[k]
                llhs = merge_llhs(llhs, ref_path, start_ids)
                print(llhs)
        print(k, llhs, file=fid)

if __name__ == '__main__':
    main()
