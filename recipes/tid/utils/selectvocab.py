'Select the set of n-grams the most relevant for topic classication.'

import argparse
from collections import defaultdict
import pickle

from ngramfeatures import select_ngrams


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
    parser.add_argument('-n', '--ngram-order', default=3, type=int,
                        help='order of the n-grams (default: 3)')
    parser.add_argument('-k', '--nbest', default=100, type=int,
                        help='number of n-grams to keep per topic (default: 100)')
    parser.add_argument('doclabels', help='list of pairs "docid topicid')
    parser.add_argument('trans', help='document transcription')
    parser.add_argument('vocab', help='output vocabulary')
    args = parser.parse_args()

    # Estimate the prior probabilities of the topics.
    doc_topic = {docid: topicid
                 for docid, topicid in iterate_doclabels(args.doclabels)}
    topic_counts = defaultdict(int)
    topics = sorted(list(set(doc_topic.values())))
    for docid, topicid in doc_topic.items():
       topic_counts[topicid] += 1
    tot = sum(topic_counts.values())
    topic_prob = [topic_counts[topic] / tot for topic in topics]

    vocab = select_ngrams(
        args.ngram_order,
        args.nbest,
        corpus=iterate_trans(args.trans),
        doc_topic=doc_topic,
        topic_prob={topicid: prob for topicid, prob in zip(topics, topic_prob)}
    )

    with open(args.vocab, 'wb') as f:
        pickle.dump(vocab, f)


if __name__ == '__main__':
    main()

