'''Compute per frame/unit level likelihood'''

import argparse
import sys
import os
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
                start_pdfs.append(int(tokens[0]))
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
    phone_llhs.append(llhs_sum / frame_count)
    return phone_llhs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', type=str, default=None,
        help='A npz file with pdf id sequence. If not empty, use it \
        to merge per farme likelihood into phone level')
    parser.add_argument('--align', type=str,
        help='Align graph file. If not empty, compute the llhs along best path.')
    parser.add_argument('--pdf2phone', type=str,
        help='File mapping pdf id(int) into phone symbols(str)')
    parser.add_argument('model', type=str, help='Decoding model')
    parser.add_argument('feats', type=str, help='Features to process')
    parser.add_argument('outdir', type=str, help='Output file')
    args = parser.parse_args()
    feats = np.load(args.feats)
    outdir = args.outdir
    smoothfile = args.smooth
    align_graphs = None
    if args.align is not None:
        align_graphs = np.load(args.align)
    mapfile = args.pdf2phone
    smooth = None

    if smoothfile:
        if not mapfile:
            sys.exit('Pdf2phone mapping file needed if merging likelihood' +
                     'into phone level')
        else:
            paths = np.load(smoothfile)
            _, start_ids = read_pdf2phone(mapfile)
            smooth = 1
    with open(args.model, 'rb') as m:
        model = pickle.load(m)

    for line in sys.stdin:
        k = line.strip()
        ft = torch.from_numpy(feats[k]).float()
        stats = model.sufficient_statistics(ft)
        if align_graphs is not None:
            graph = align_graphs[k][0]
            llhs = model.expected_log_likelihood(stats,
                                                 inference_graph=graph).tolist()
        else:
            llhs = model.expected_log_likelihood(stats).tolist()
        if smooth:
            ref_path = paths[k]
            llhs = merge_llhs(llhs, ref_path, start_ids)
        outfile = os.path.join(outdir, k + '.npy')
        np.save(outfile, llhs)

if __name__ == '__main__':
    main()
