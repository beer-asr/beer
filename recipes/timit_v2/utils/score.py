
import argparse
import numpy as np
import sys



def main():
    parser = argparse.ArgumentParser(description='Score with DTW method')
    parser.add_argument('reference', help='npz file with reference in integers')
    parser.add_argument('hypothesis', help='npz file with reference in intergers')
    args = parser.parse_args()

    all_ref = np.load(args.reference)
    all_hyp = np.load(args.hypothesis)

    ref_keys = list(all_ref.keys())
    hyp_keys = list(all_hyp.keys())

    if not sorted(ref_keys) == sorted(hyp_keys):
        sys.exit('Reference and hypothesis do not have same utterance ids')

    tot_len = 0
    tot_err = 0
    for k in ref_keys:
        ref = all_ref[k]
        hyp = all_hyp[k]
        mtrix = np.zeros((len(hyp) + 1, len(ref) + 1))
        for i in range(len(ref) + 1):
            mtrix[0, i] = i
        for i in range(len(hyp) + 1):
            mtrix[i, 0] = i
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                ins_err = mtrix[j-1, i] + 1
                del_err = mtrix[j, i-1] + 1
                sub_err = mtrix[j-1, i-1] + int(ref[i - 1] != hyp[j - 1])
                err = [ins_err, del_err, sub_err]
                mtrix[j, i] = min(err)
        tot_err += mtrix[-1, -1]
        tot_len += len(ref)

    per = 100 * tot_err / tot_len
    print('Phone Error Rate:', round(per, 3), '%')

if __name__ == "__main__":
    main()
