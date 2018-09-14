
import argparse
import numpy as np
import sys

def read_text(fid):
    dict_utt = {}
    with open(fid, 'r') as f:
        for line in f:
            line = line.strip().split()
            uttid = line.pop(0)
            dict_utt[uttid] = line
    return dict_utt

def read_phone_map(phone_map):
    '''For TIMIT only: read 48-to-38 phoneme map file
    '''
    if phone_map:
        dict_phone_map = {}
        with open(phone_map, 'r') as f:
            for line in f:
                line = line.strip().split()
                dict_phone_map[line[0]] = line[1]
        return dict_phone_map


def filter_text(text, remove=[0], duplicate='no', phone_map=None):
    '''Remove certain units; remove adjacent duplicated units; (For TIMIT only)
        convert 48 phonemes into 39 phonemes
    '''
    if phone_map is not None:
       if remove is not None:
           remove = remove.split()
           filter_text = [phone_map[i] for i in text if (i not in remove) and
                          phone_map[i] not in remove]
       else:
           filter_text = [phone_map[i] for i in text]
    else:
        if remove is not None:
            remove = remove.split()
            filter_text = [i for i in text if i not in remove]
        else:
            filter_text = text
    if duplicate == 'no':
        new_text = [elem for i, elem in enumerate(filter_text)
                    if i == 0 or filter_text[i-1] != elem]
    else:
        new_text = filter_text
    return new_text

def main():
    parser = argparse.ArgumentParser(description='Score with DTW method')
    parser.add_argument('--remove', default=None,
        help='Units to be removed when scoring')
    parser.add_argument('--duplicate', default='yes', type=str,
        help='Allow same phone to duplicate in sequence')
    parser.add_argument('--phone_map', help='For TIMIT only: 48 to 39 phoneme map')
    parser.add_argument('reference', help='Transcription file')
    parser.add_argument('hypothesis', help='Decoded hypothesis result file')
    args = parser.parse_args()

    all_ref = read_text(args.reference)
    all_hyp = read_text(args.hypothesis)
    if args.remove is None:
        remove_unit = None
    else:
        remove_unit = args.remove

    duplicate = args.duplicate
    phone_map = read_phone_map(args.phone_map)
    ref_keys = list(all_ref.keys())
    hyp_keys = list(all_hyp.keys())

    if not sorted(ref_keys) == sorted(hyp_keys):
        sys.exit('Reference and hypothesis do not have same utterance ids')

    tot_len = 0
    tot_err = 0
    for k in sys.stdin:
        k = k.strip()
        ref = filter_text(all_ref[k], remove=remove_unit, duplicate=duplicate,
            phone_map=phone_map)
        hyp = filter_text(all_hyp[k], remove=remove_unit, duplicate=duplicate,
            phone_map=phone_map)
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
        print(k, round(mtrix[-1, -1] / len(ref), 4))

    #per = tot_err / tot_len
    #print('Total: ', round(per, 4))

if __name__ == "__main__":
    main()
