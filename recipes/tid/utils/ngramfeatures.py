'''Implement the features extraction of text document maximizing the
mutual information w.r.t. between the documents and the topic labels.

'''

from collections import defaultdict


class NGramCounter:
    '''Help to count the n-gram sequentially from a set of sentences.

    Attributes:
        order (int): n-gram order to consider
        counts (dict): number of occurences for each n-gram
        prior_count (float): pseudo number count assigned to each
            n-gram before seeing any data.

    Notes:
        * The n-gram count are stored in floating point precision as
          the user may give "prior count"
        * setting the "prior_count" attribute will reset the counts for
          all the n-gram

    Example:
        Basic usage:
        >>> counter = NGramCounter(ngram_order=2)
        >>> counter.add('this is a first sentence')
        >>> for key, val in counter.counts.items():
        ...     print(f'ngram "{key}" has {val:.2f} counts')
        ngram "('<s>', 'this')" has 1.00 counts
        ngram "('this', 'is')" has 1.00 counts
        ngram "('is', 'a')" has 1.00 counts
        ngram "('a', 'first')" has 1.00 counts

        Call the method "add" again to accumulate the counts
        >>> counter.add('this is a second sentence')
        >>> for key, val in counter.counts.items():
        ...     print(f'ngram "{key}" has {val:.2f} counts')
        ngram "('<s>', 'this')" has 2.00 counts
        ngram "('this', 'is')" has 2.00 counts
        ngram "('is', 'a')" has 2.00 counts
        ngram "('a', 'first')" has 1.00 counts
        ngram "('a', 'second')" has 1.00 counts

        Setting a prior count
        >>> counter = NGramCounter(ngram_order=2, prior_count=.5)
        >>> counter.add('this is a first sentence')
        >>> for key, val in counter.counts.items():
        ...     print(f'ngram "{key}" has {val:.2f} counts')
        ngram "('<s>', 'this')" has 1.50 counts
        ngram "('this', 'is')" has 1.50 counts
        ngram "('is', 'a')" has 1.50 counts
        ngram "('a', 'first')" has 1.50 counts

    '''

    def __init__(self, ngram_order=3, prior_count=0.):
        self._prior_count = prior_count
        self.order = ngram_order
        self.counts = defaultdict(lambda: float(prior_count))

    @property
    def prior_count(self):
        return self._prior_count

    @prior_count.setter
    def prior_count(self, value):
        self._prior_count = value
        self.counts = defaultdict(lambda: float(value))

    def add(self, sentence):
        '''Count the n-gram of the given document and accumulate it to
        the global counter.

        Note:
            Input sentences are left padded with the symbol
            "<s>" such that there are as many n-gram
            as the number of word in the sentence.

        '''
        padded_sentence = ' '.join(['<s> ' * (self.order - 1), sentence])
        tokens = padded_sentence.split()
        for i in range(len(tokens) - self.order):
            self.counts[tuple(tokens[i:i+self.order])] += 1

    def get_counts(self, vocab):
        '''Returns the counts for the element of the given vocabulary.

        Args:
            vocab (list): List of n-grams for which to get the counts.

        Returns:
            list of counts

        Example:
            >>> counter = NGramCounter(ngram_order=2)
            >>> counter.add('a b c a b d')
            >>> counter.get_counts([('a', 'b'), ('b', 'c')])
            [2.0, 1.0]

        '''
        return [self.counts[ngram] for ngram in vocab]


def select_ngrams(ngram_order, nbest, corpus, doc_topic, topic_prob):
    '''Select the n-grams which are the most relevant to the
    topic classification task.

    Args:
        ngram_order (int): Order of the n-gram.
        nbest (int): Number of n-gram per topic to keep.
        corpus (iterable): Iterable over pairs (docid, sentence).
        doc_topic (dict): Mapping document id -> topic id.
        topic_prob (dict): Prior probability for each topic.

    Returns:
        a list of n-gram stored as tuples.

    Example:
        >>> corpus = [
        ...     ('doc1', 'this is great'),
        ...     ('doc2', 'this is cool'),
        ...     ('doc3', 'that is great'),
        ...     ('doc4', 'that is cool')
        ... ]
        >>> topic_prob = {'topic1': 0.3, 'topic2': 0.7}
        >>> doc_topic = {'doc1': 'topic1', 'doc2': 'topic1',
        ...              'doc3': 'topic2', 'doc4': 'topic2'}
        >>> select_ngrams(ngram_order=2, nbest=1, corpus=corpus,
        ...               doc_topic=doc_topic, topic_prob=topic_prob)
        [('that', 'is'), ('this', 'is')]

    '''
    # Count the n-gram per topic
    global_counter = NGramCounter(ngram_order, prior_count=len(topic_prob))
    topic_counters = {topicid: NGramCounter(ngram_order,
                                            prior_count=len(topic_prob) * p_t)
                      for topicid, p_t in topic_prob.items()}
    for docid, sentence in corpus:
        topicid = doc_topic[docid]
        topic_counters[topicid].add(sentence)
        global_counter.add(sentence)

    # Ranked the n-gram per topic.
    topic_ranked_ngrams = {}
    for topic in topic_prob:
        ranked_ngrams = []
        for ngram in global_counter.counts:
            score = topic_counters[topic].counts[ngram] / \
                        global_counter.counts[ngram]
            ranked_ngrams.append((ngram, score))
        topic_ranked_ngrams[topic] = list(reversed(sorted(ranked_ngrams,
                                     key=lambda x: x[1])))

    # Get the vocabulary by merging the n-gram.
    vocab = set()
    for ngrams in topic_ranked_ngrams.values():
        vocab = vocab.union({ngram for ngram, _ in ngrams[:nbest]})

    return sorted(list(vocab))


if __name__ == '__main__':
    import doctest
    print(doctest.testmod())

