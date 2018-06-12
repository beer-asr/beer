
''' HMM model training
'''
import random
import numpy as np
import torch
import argparse
import sys
sys.path.insert(0, '../../beer')
import beer
import pickle
import logging

def read_phones(phonefile):
    '''Read phonemes file
    '''
    logging.info("Reading phone list")
    with open(phonefile, 'r') as p:
        phonelist = [l.rstrip('\n') for l in p]
    phones_dict = {s.split()[0]: int(s.split()[1]) for s in phonelist}

    return phones_dict

def feature_transform(feat, mean_norm='False', var_norm='False', mean=None,
                      std=None, context=0):

    '''Perform feature transformation with mean/var normalization, and append
    context
    Args:
    feat (np.array(float)): feature
    mean_norm(str): True or False
    var_norm(str): True or False
    mean(np.array(float)): Global mean
    std(np.array(float)): Global standard deviation
    context(int): context frame for both left and right sides
    '''
    if var_norm == 'True':
        if (std is None) or (mean is None):
            sys.exit('Global standard deviation or mean is not given \
                     while performing variance normalization')
        else:
            feat = (feat - mean) / std + mean
    if mean_norm == 'True':
        if mean is None:
            sys.exit('Global mean is not given while performing mean normalization')
        else:
            feat -= mean
     #xiao_xiong_has_made_a_bug = None
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
    parser.add_argument('feats', type=str, help='Feature file')
    parser.add_argument('labels', type=str, help='Label file')
    parser.add_argument('emissions', type=str, help='Emission modelset file')
    parser.add_argument('feat_stats', type=str, default=None,
        help='Feature statistics: mean, standard deviation, count')
    parser.add_argument('hmm_model_dir', type=str, help='Output trained HMM model')
    parser.add_argument('--mean_normalize', default='True',
        choices=['True', 'False'])
    parser.add_argument('--var_normalize', default='False',
        choices=['True', 'False'])
    parser.add_argument('--context', type=int, default=5,
        help='Length of feature for both sides')
    parser.add_argument('--lrate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int,
        help='Utterance number in each batch')
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()



    # Read arguments
    feats = np.load(args.feats)
    labels = np.load(args.labels)
    hmm_mdl_dir = args.hmm_model_dir
    stats = np.load(args.feat_stats)
    mean_norm = args.mean_normalize
    var_norm = args.var_normalize
    context = args.context
    lrate = args.lrate
    batch_size = args.batch_size
    epochs = args.epochs
    with open(args.emissions, 'rb') as pickle_file:
        emissions = pickle.load(pickle_file)

    global_mean = stats['mean']
    global_std = stats['std']
    tot_counts = int(stats['count'])

    log_format = "%(asctime)s :%(lineno)d %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Prepare data
    keys = list(feats.keys())
    random.shuffle(keys)
    batches = [keys[i: i+batch_size] for i in range(0, len(keys), batch_size)]
    logging.info("Data shuffled into %d batches", len(batches))

    # Training
    elbo_fn = beer.EvidenceLowerBound(tot_counts)
    params = emissions.parameters
    optimizer = beer.BayesianModelOptimizer(params, lrate)

    for epoch in range(epochs):
        logging.info("Epoch: %d", epoch)
        hmm_epoch = hmm_mdl_dir + '/' + str(epoch) + '.mdl'
        for batch_keys in batches:
            optimizer.zero_grad()
            for utt in batch_keys:
                logging.info("Training with utterance %s", utt)
                ft = feature_transform(feats[utt], mean_norm=mean_norm,
                     var_norm=var_norm, mean=global_mean, std=global_std,
                     context=context)
                lab = labels[utt]
                init_state = torch.tensor([0])
                final_state = torch.tensor([len(lab) - 1])
                trans_mat_ali = beer.HMM.create_ali_trans_mat(len(lab))
                ali_sets = beer.AlignModelSet(emissions, lab)
                hmm_ali = beer.HMM.create(init_state, final_state,
                          trans_mat_ali, ali_sets)
                elbo = elbo_fn(hmm_ali, torch.from_numpy(ft).float())
                #elbo.natural_backward()
            logging.info("Elbo value is %f", float(elbo) / tot_counts)
            #optimizer.step()
        with open(hmm_epoch, 'wb') as m:
            pickle.dump(emissions, m)

    hmm_mdl = hmm_mdl_dir + '/hmm.mdl'
    with open(hmm_mdl, 'wb') as m:
        pickle.dump(emissions, m)

if __name__ == "__main__":
    main()
