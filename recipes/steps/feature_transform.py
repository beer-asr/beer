
'''Perform feature transformation including mean/variance normalization 
    on global or per utterance base; adding delta/double delta to features;
    concanating context to each frame.   
'''
import numpy as np
import argparse
import beer
from accumulate_data_stats import accumulate
import sys

def feature_transform(feat, context=0, mean=None, std=None, 
                      mean_norm=None, var_norm=None, add_delta=None):

    '''Perform feature transformation with mean/var normalization, and append
    context
    Args:
        feat (np.array(float)): feature
        context(int): context frame for both left and right sides
        mean: None or np.array(float) when mean_norm or var_norm required
        std: None or np.array(float) when var_norm required
        mean_norm: if not None, it should be the mean of feature(np.array(float))
        var_norm(str): if not None, it should be the std of
                       feature(np.array(float))
        add_delta: None or str
    Returns:
        feat: np.array(float)
    '''
    if var_norm :
        if (mean is None) or (std is None):
            sys.exit('Mean or standard deviation is not given when perform \
                      variance normalization !')
        else:
            feat = (feat - mean) / std + mean
    if mean_norm:
        if mean is None:
            sys.exit('Mean is not given while performing mean normalization !')
        else:
            feat -= mean
     #xiao_xiong_has_made_a_bug = None
    if add_delta:
        feat = beer.features.add_deltas(feat)

    if context != 0:
        padded = np.r_[np.repeat(feat[0][None], context, axis=0),
                           feat, np.repeat(feat[-1][None], context, axis=0)]
        feat_stacked = np.zeros((len(feat), (2 * context + 1) * feat.shape[1]))
        for i in range(context, len(feat) + context):
            fea = padded[(i - context) : (i + context + 1)]
            feat_stacked[i - context] = fea.reshape(-1)
        feat = feat_stacked

    return feat

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('original_feats_dir', type=str)
    parser.add_argument('--mean-norm', action='store_true',
        help='Mean normalization')
    parser.add_argument('--var-norm', action='store_true',
        help='Variance normalization')
    parser.add_argument('--norm_type', default='per_utt',
        choices=['per_utt', 'global'])
    parser.add_argument('--add-delta', action='store_true',
        help='Add delta and double delta')
    parser.add_argument('--context', type=int, default=0,
        help='Length of feature context on both left and right sides')
    args = parser.parse_args()

    ori_feats = args.original_feats_dir + '/feats.npz'
    mean_norm = args.mean_norm
    var_norm = args.var_norm
    norm_type = args.norm_type
    add_delta = args.add_delta
    context = args.context
    tmpdir = args.original_feats_dir + '/feat_transform/'
    
    global_mean, global_std, _, dict_utt_details = accumulate(ori_feats)
    ori_feats = np.load(ori_feats)
    if norm_type == 'per_utt':
        for utt in ori_feats.keys():
            ft = feature_transform(ori_feats[utt], context,
                 mean=dict_utt_details[utt][0],
                 std=dict_utt_details[utt][1],
                 mean_norm=mean_norm, var_norm=var_norm, add_delta=add_delta)
            np.save(tmpdir + utt + '.npy', ft)
    else:
        for utt in ori_feats.keys():
            ft = feature_transform(ori_feats[utt], context,
                 mean=global_mean, std=global_std,
                 mean_norm=mean_norm, var_norm=var_norm, add_delta=add_delta)
            np.save(tmpdir + utt + '.npy', ft)

if __name__ == '__main__':
    main()
