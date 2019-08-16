'Extract bag-of-word representation from a transcription.'

import argparse
from collections import defaultdict
import pickle

import numpy as np

from ngramfeatures import NGramCounter


def iterate_doclabels(path):
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            yield tokens[0], ' '.join(tokens[1:])


def iterate_trans(path):
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            yield tokens[0], ' '.join(tokens[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('doclabels', help='list of pairs "docid topicid')
    parser.add_argument('trans', help='document transcription')
    parser.add_argument('vocab', help='vocabulary to use for the bag-of-words')
    parser.add_argument('fea', help='output features (".npz" will be added) ')
    args = parser.parse_args()

    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)

    doc_topic = {docid: topicid for docid, topicid in iterate_doclabels(args.doclabels)}
    topic_counts = defaultdict(int)
    topics = sorted(list(set(doc_topic.values())))

    docs = defaultdict(list)
    for docid, sentence in iterate_trans(args.trans): docs[docid].append(sentence)

    X = np.zeros((len(docs), len(vocab)), dtype=float)
    y = np.zeros(len(docs))
    for i, docid in enumerate(docs):
        counter = NGramCounter(ngram_order=len(vocab[0]))
        for sentence in docs[docid]: counter.add(sentence)
        X[i] = counter.get_counts(vocab)
        y[i] = topics.index(doc_topic[docid])

    np.savez(args.fea, fea=X, labels=y)


if __name__ == '__main__':
    main()

